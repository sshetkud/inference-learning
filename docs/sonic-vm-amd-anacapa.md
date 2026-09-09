# SONiC VM on KVM — AMD `amd_anacapa` (MiniRack AFM)

Runbook for creating a SONiC VM on a Conductor GPU node using the **ONIE embed + `sonic-broadcom.bin` install** flow from the AMD MiniRack AFM bring-up wiki.

Validated on **smc200x-ccs-e12-43** (`sonic-vm2`, SONIC-128 `202505_1.0.0-128`).

---

## Host & constraints

| Item | Value |
|------|--------|
| Example host | `smc200x-ccs-e12-43.cs-aus.dcgpu` |
| Work dir | `/etc/aifm/sonic-vm/` |
| **Do NOT bridge** | `ens238f0` (mgmt NIC — breaks SSH) |
| VM network | libvirt `default` (`virbr0` = `192.168.122.1`) |
| SONiC VM IP | DHCP on `virbr0` (e.g. `192.168.122.177` — check with `virsh domifaddr`) |
| AFM package (host) | `/etc/aifm/afm-package/` |
| Console | `sudo virsh console <vm>` |
| Install disk | **32 GB** NVMe qcow2 minimum (8 GB fails first boot) |
| NIC model | **e1000** (virtio not visible in ONIE) |
| Machine | `pc` (i440fx) |
| Install target | emulated **NVMe** (`/dev/nvme0n1`) via qemu commandline |

---

## Artifacts (Artifactory)

| File | Path |
|------|------|
| ONIE recovery ISO (KVM) | `SONIC-ONIE-ISO/onie-recovery-x86_64-amd_anacapa-r0_smallerdisk.iso` |
| SONiC installer | `SONIC-128/sonic-broadcom.bin` (or `SONIC-BUILD-133/…`) |

```bash
cd /etc/aifm/sonic-vm
sudo mkdir -p SONIC-128

sudo curl -kfsSL -o onie-recovery-x86_64-amd_anacapa-r0_smallerdisk.iso \
  "https://mkmartifactory.amd.com:8443/artifactory/SW-AMDNOS-SONIC-LOCAL/SONIC-ONIE-ISO/onie-recovery-x86_64-amd_anacapa-r0_smallerdisk.iso"

sudo curl -kfsSL -o SONIC-128/sonic-broadcom.bin \
  "https://mkmartifactory.amd.com/artifactory/SW-AMDNOS-SONIC-LOCAL/SONIC-128/sonic-broadcom.bin"
```

---

## Phase 0 — One-time host setup

```bash
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients virtinst qemu-utils curl

# AppArmor / libvirt fix
grep -q '^security_driver = "none"' /etc/libvirt/qemu.conf || \
  echo 'security_driver = "none"' | sudo tee -a /etc/libvirt/qemu.conf
sudo systemctl restart libvirtd

sudo mkdir -p /etc/aifm/sonic-vm /var/lib/libvirt/images/sonic
```

---

## Phase 1 — Create VM & embed ONIE

Creates a **32 GB** blank NVMe disk and boots the **ONIE recovery ISO**.

```bash
git clone https://github.com/sshetkud/inference-learning.git
cd inference-learning/scripts/sonic-vm
sudo bash recreate_sonic_vm2_32g.sh
# Optional: VM=sonic-vm1 DISK=/var/lib/libvirt/images/sonic/sonic-vm1-disk.qcow2 sudo -E bash recreate_sonic_vm2_32g.sh
```

```bash
sudo virsh console sonic-vm2   # Ctrl+] to exit
```

Wait for embed success:

```
Installing ONIE on: /dev/nvme0n1
ONIE: Success: Firmware update version: master-02121638.0.1
ONIE: Rebooting...
```

---

## Phase 2 — Install SONiC (HTTP)

**CD-ROM hotplug does not work** on this VM type. Stop the VM, redefine XML (HD boot only, e1000 NIC), start HTTP server, then install from ONIE.

```bash
sudo bash sonic-vm2-install-32g.sh
```

Starts HTTP on `192.168.122.1:8081` serving `sonic-broadcom.bin`.

```bash
sudo virsh console sonic-vm2
```

At `ONIE:/ #`:

```bash
ip link set eth0 up
udhcpc -i eth0
onie-nos-install http://192.168.122.1:8081/sonic-broadcom.bin
```

Wait for download → checksum → install → reboot (~10–20 min).

> Use an `http://` URL. Bare `/dev/sr0` fails with `Unknown URL type` on this ONIE build.

---

## Phase 3 — Login & verify

```
login: admin
password: YourPaSsWoRd
```

SONIC-128 ships a **custom** image password. If login fails, reset from the host:

```bash
sudo bash reset_sonic_admin.sh
```

Inside SONiC:

```bash
sonic-installer list
show version
```

---

## Phase 4 — Upgrade (optional)

Only when SONiC is **already running** (not from ONIE):

```bash
sudo sonic-installer install sonic-broadcom.bin --skip_migration
sudo sonic-installer list
sudo reboot
```

---

## Phase 5 — AFM controller bring-up (on SONiC VM)

Maps to **AFM Bring-Up wiki Steps 3–7** for a single-switch **gpu8** lab. The controller runs **inside the SONiC VM** (not on the KVM host).

All `config` / `docker` / `bootstrap.py` commands below run **on SONiC** (`admin@sonic`), via console or SSH:

```bash
# From KVM host — console
sudo virsh console sonic-vm2

# Or SSH (password: YourPaSsWoRd)
SONIC_IP=$(sudo virsh domifaddr sonic-vm2 | awk '/192\.168\.122/{print $4}' | cut -d/ -f1)
ssh admin@${SONIC_IP}
```

### 5.0 — Stage AFM images on SONiC

If images are not already under `/etc/afm/agent/afm-package/images/` (or similar), copy from the host:

```bash
# On KVM host (e12-43)
AFM_PKG=/etc/aifm/afm-package
SONIC_IP=$(sudo virsh domifaddr sonic-vm2 | awk '/192\.168\.122/{print $4}' | cut -d/ -f1)

scp ${AFM_PKG}/images/afm_controller_container.tar admin@${SONIC_IP}:/tmp/
scp ${AFM_PKG}/images/afm_agent_container.tar     admin@${SONIC_IP}:/tmp/   # optional, for agent

# On SONiC
sudo mkdir -p /etc/afm/agent/afm-package/images
sudo mv /tmp/afm_controller_container.tar /etc/afm/agent/afm-package/images/
cd /etc/afm/agent/afm-package/images
```

### 5.1 — Load & tag controller image (wiki Steps 3–4)

```bash
cd /etc/afm/agent/afm-package/images

sudo docker load -i afm_controller_container.tar
sudo docker tag pen-afm-controller:latest docker-afm-controller:latest
sudo docker images | grep -i afm
```

### 5.2 — Controller coordinates (wiki Step 5)

Set coordinates to the **SONiC switch management IP** (not the KVM host). The VM gets DHCP from `virbr0`:

```bash
# On SONiC — confirm eth0 IP
ip -4 addr show eth0

# Example: 192.168.122.177/24
sudo config controller coordinates 192.168.122.177
sudo config save -y
```

| IP | Role |
|----|------|
| `192.168.122.177` | SONiC VM (`eth0`) — use for **coordinates** and **bootstrap AFM_IP** |
| `192.168.122.1` | KVM host (`virbr0`) — reachability only; not coordinates in this layout |

> `sudo config controller coordinates …` on the **Linux host** fails with `config: command not found`. It is a **SONiC CLI** command only.

### 5.3 — Start `afm-controller` (wiki Step 6)

```bash
sudo systemctl start afm-controller
sudo docker ps | grep afm-controller
```

If `systemctl` does not create a container (`No such container: afm-controller`), start manually:

```bash
sudo docker rm -f afm-controller 2>/dev/null
sudo docker run -d \
  --name afm-controller \
  --network host \
  --privileged \
  --restart unless-stopped \
  -v /opt/amd/afm:/opt/amd/afm \
  docker-afm-controller:latest

# Wait for controller process (~15 s)
sleep 15
sudo docker logs afm-controller --tail 20
```

Expect: `afm_controller is running (PID: …)`.

### 5.4 — Bootstrap cluster (wiki Step 7)

Replace `<sw_mgmt_ip>` with the SONiC VM IP from step 5.2 (e.g. `192.168.122.177`).

```bash
sudo docker exec afm-controller python3 /controller_pkg/bootstrap.py \
  -clustername aifm-cluster \
  -password 'Pensando0$' \
  -rack_type gpu8 \
  -output_log bootstrap.log \
  192.168.122.177
```

**Bootstrap is quiet for several minutes** — that is normal. Progress is written to `bootstrap.log` inside the container:

```bash
# In another SONiC session while bootstrap runs
sudo docker exec afm-controller tail -f bootstrap.log
```

Successful completion looks like:

```
* AFM bootstrap completed successfully
* Created pod: pod-1 Rack type: gpu8 IFCP encryption mode: disabled
* you may access AFM at https://192.168.122.177
```

A single `409` retry (`attempt #1 received response code: 409, retrying...`) is harmless.

Default UI login: **`admin`** / **`Pensando0$`**.

### 5.5 — Verify controller

```bash
sudo docker ps | grep afm-controller
curl -sk -o /dev/null -w "HTTPS:%{http_code}\n" https://192.168.122.177/
```

---

## Phase 6 — AFM agent (switch)

If the agent image is loaded and `afm-agent` is not already running:

```bash
cd /etc/afm/agent/afm-package/images
sudo docker load -i afm_agent_container.tar   # if needed

sudo systemctl start afm-agent
sudo docker ps | grep afm-agent
```

Compute-node agents are deployed separately (see full **AFM Bring-Up** wiki / `deploy_afm.sh` with `[compute]` inventory).

---

## Phase 7 — Access AFM UI from your laptop

SONiC is on the private `192.168.122.0/24` network. Tunnel HTTPS through the Conductor node:

```bash
# Terminal 1 — leave open (-N = tunnel only; blank screen is expected)
ssh -N -L 8443:192.168.122.177:443 sshetkud@smc200x-ccs-e12-43.cs-aus.dcgpu
```

Open **https://localhost:8443** in a browser (accept the self-signed certificate).

Background tunnel:

```bash
ssh -f -N -L 8443:192.168.122.177:443 sshetkud@smc200x-ccs-e12-43.cs-aus.dcgpu
```

Replace `192.168.122.177` if `virsh domifaddr` shows a different address.

---

## VM XML essentials

**Phase 1 (embed):** boot `cdrom` then `hd`; CD = ONIE recovery ISO.

**Phase 2 (install):** boot `hd` only; recovery ISO removed.

**Both phases:**

```xml
<interface type='network'>
  <source network='default'/>
  <model type='e1000'/>
</interface>
```

NVMe via qemu commandline (not libvirt `bus=nvme`):

```xml
<qemu:commandline xmlns:qemu='http://libvirt.org/schemas/domain/qemu/1.0'>
  <qemu:arg value='-drive'/>
  <qemu:arg value='file=/var/lib/libvirt/images/sonic/sonic-vm2-disk.qcow2,format=qcow2,if=none,id=NVME0'/>
  <qemu:arg value='-device'/>
  <qemu:arg value='nvme,drive=NVME0,serial=nvme-0'/>
</qemu:commandline>
```

---

## Pitfalls

| Wrong | Right |
|-------|-------|
| `sonic-vs.qcow2` community image | `sonic-broadcom.bin` + ONIE embed |
| 8 GB disk | **32 GB** disk |
| virtio NIC | **e1000** NIC |
| virtio/SATA install disk | **NVMe** via qemu `-device nvme` |
| Hotplug CD while VM runs | Cold attach at `virsh define` |
| `onie-nos-install /dev/sr0` | `onie-nos-install http://192.168.122.1:8081/...` |
| `sonic-installer` from ONIE | `onie-nos-install` (first install) |
| Bridge `ens238f0` | libvirt `default` network |
| Keep recovery ISO after embed | Remove ISO; boot HD only |
| `config` on Linux host | `config` only on **SONiC** shell |
| `docker load` only | Still need `systemctl start afm-controller` or `docker run` |
| Coordinates = host `10.235.x.x` | Coordinates = **SONiC VM IP** on `virbr0` (e.g. `192.168.122.177`) |
| Bootstrap prints nothing | Normal — tail `bootstrap.log` inside container (~3–5 min) |
| `ssh -L` “hangs” | Expected — tunnel stays open; browse in another terminal |

---

## Helper scripts

In [scripts/sonic-vm/](../scripts/sonic-vm/):

| Script | Purpose |
|--------|---------|
| `recreate_sonic_vm2_32g.sh` | Phase 1: 32G disk + ONIE embed boot |
| `sonic-vm2-install-32g.sh` | Phase 2: HD boot + HTTP install |
| `reset_sonic_admin.sh` | Reset `admin` → `YourPaSsWoRd` (offline disk edit) |

Environment overrides (all scripts): `VM`, `DISK`, `BIN`, `HOST_IP`, `HTTP_PORT`, `PASS`.

---

## Related wiki

[AFM Bring-Up](https://amd.atlassian.net/wiki/spaces/DCGPUCEVAL/pages/1810969287/AFM+Bring-Up) (Confluence) — full rack procedure. This runbook covers:

| Wiki scope | This doc |
|------------|----------|
| Physical switch / SONiC install | Phases 0–4 (KVM + `amd_anacapa` ONIE) |
| AFM controller Steps 3–7 | Phases 5–7 (controller on SONiC VM) |
| Multi-switch `deploy_afm.sh` | Use package inventory with `[switch]` / `[compute]` groups |
