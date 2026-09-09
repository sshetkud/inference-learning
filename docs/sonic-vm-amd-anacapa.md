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

AMD MiniRack E2E AFM Bring-Up (Confluence) — Steps 1–5 for physical switch; this runbook adapts Step 1 for KVM with `amd_anacapa` ONIE.
