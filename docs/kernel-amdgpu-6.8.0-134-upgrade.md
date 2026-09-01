# Kernel 6.8.0-134 + AMDGPU / AINIC DKMS Upgrade (MI355X r17)

Runbook for upgrading Conductor GPU nodes from `6.8.0-111-generic` to `6.8.0-134-generic` with matching DKMS modules for GPU (amdgpu) and AINIC (ionic, pds, tawk-ipc).

Validated on **smci355-ccs-aus-r17-34** (MI355X, ionic/AINIC).

---

## Prerequisites

- Root/sudo on the node
- Maintenance window (reboot required)
- For 2-node RCCL: repeat on **both** nodes and align kernel + driver versions

---

## Phase 0 — Baseline

```bash
hostname -s
uname -r
rocm-smi --showdriverversion
dpkg -l | grep -iE 'amdgpu-dkms|linux-image'
ls -d /usr/src/amdgpu-* 2>/dev/null
dkms status | grep amdgpu
```

Example reference (**r17-22**, before upgrade):

| Item | Value |
|------|--------|
| Kernel | `6.8.0-111-generic` |
| amdgpu DKMS | `6.16.13-2303411.24.04` |
| Driver | `30.30.01` |

---

## Phase 1 — Install kernel 6.8.0-134

```bash
sudo apt update
sudo apt install \
  linux-image-6.8.0-134-generic \
  linux-headers-6.8.0-134-generic \
  linux-modules-6.8.0-134-generic \
  linux-modules-extra-6.8.0-134-generic
```

### Confirm headers path exists

```bash
ls -d /lib/modules/6.8.0-134-generic/build
ls /boot/vmlinuz-6.8.0-134-generic
```

Should point to `/usr/src/linux-headers-6.8.0-134-generic`.

### If packages are not found

```bash
apt-cache policy linux-headers-6.8.0-134-generic
apt-cache policy linux-image-6.8.0-134-generic
```

| Result | Action |
|--------|--------|
| Package not found | Kernel 134 is not in apt repos — ask admin to add it or refresh the cluster node image |
| Only image, no headers | Mirror/repo incomplete — admin fix |
| Different version available | Use the kernel version your site actually ships |

---

## Phase 2 — Ensure amdgpu-dkms source exists

```bash
dpkg -l | grep -i amdgpu-dkms
ls -d /usr/src/amdgpu-* 2>/dev/null
```

| Result | Action |
|--------|--------|
| No `/usr/src/amdgpu-*` | `sudo apt install amdgpu-dkms` |
| Package installed, source missing | `sudo apt install --reinstall amdgpu-dkms` |
| apt fails | Check AMD repo: `cat /etc/apt/sources.list.d/*amdgpu*` — escalate to admin |

**Important:** The DKMS version string differs by node. Read yours from `/usr/src/` — do not assume `2303411`.

```bash
ls -d /usr/src/amdgpu-*
# r17-22 example: /usr/src/amdgpu-6.16.13-2303411.24.04
# r17-34 example: /usr/src/amdgpu-6.16.13-2317211.24.04
```

Set a variable for Phase 3:

```bash
AMDGPU_VER=$(basename /usr/src/amdgpu-* | sed 's/amdgpu-//')
echo "Using amdgpu DKMS version: $AMDGPU_VER"
```

---

## Phase 3 — Build DKMS modules for 6.8.0-134

### Option A — explicit build (recommended)

```bash
sudo dkms install amdgpu/${AMDGPU_VER} -k 6.8.0-134-generic
```

Example (r17-34):

```bash
sudo dkms install amdgpu/6.16.13-2317211.24.04 -k 6.8.0-134-generic
```

### Option B — reinstall amdgpu-dkms (builds all installed kernels)

```bash
sudo apt install --reinstall linux-headers-6.8.0-134-generic
sudo apt install --reinstall amdgpu-dkms
sudo dkms status | grep amdgpu
```

### Common errors

| Error | Cause | Fix |
|-------|--------|-----|
| `kernel headers ... cannot be found` | Missing `linux-headers-6.8.0-134-generic` | Phase 1 — install headers |
| `Could not find module source directory` | Missing `/usr/src/amdgpu-*` | Phase 2 — install/reinstall `amdgpu-dkms` |
| Wrong version in `dkms install` | Hardcoded old version | Use `$AMDGPU_VER` from `/usr/src/` |

---

## Phase 4 — Verify DKMS (before reboot)

```bash
dkms status | grep 134
ls /lib/modules/6.8.0-134-generic/updates/dkms/amdgpu.ko*
```

**Expected on MI355X r17 with AINIC** (r17-34 success):

```
amdgpu/6.16.13-2317211.24.04, 6.8.0-134-generic, x86_64: installed
ionic/26.03.3.001,             6.8.0-134-generic, x86_64: installed
pds/1.117.5.a.77,              6.8.0-134-generic, x86_64: installed
tawk-ipc/1.117.5.a.77,         6.8.0-134-generic, x86_64: installed
```

Minimum for GPU:

```
amdgpu/<version>, 6.8.0-134-generic, x86_64: installed
```

**Do not reboot until required modules show `installed`.**

---

## Phase 5 — Set boot kernel and reboot

```bash
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-134-generic"
sudo update-grub
sudo reboot
```

---

## Phase 6 — Post-reboot verification

```bash
uname -r
# expect: 6.8.0-134-generic

rocm-smi --showdriverversion
rocm-smi

dkms status | grep 134
ibv_devinfo -l
modinfo ionic | grep ^version
```

---

## Phase 7 — Align peer node (2-node RCCL)

Both nodes must match before running RCCL:

```bash
echo "kernel: $(uname -r)"
echo "driver: $(rocm-smi --showdriverversion 2>/dev/null | tail -1)"
dkms status | grep 'amdgpu.*134'
```

| OK | Not OK |
|----|--------|
| Both `6.8.0-134-generic` | One on 111, one on 134 |
| Same driver (e.g. both 30.30.02) | Mixed 30.30.01 / 30.30.02 |
| amdgpu + ionic + pds installed for 134 | Missing modules on one node |

---

## Rollback

If GPU fails after boot to 134, boot back to 6.8.0-111:

```bash
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-111-generic"
sudo update-grub
sudo reboot
```

---

## Quick copy-paste (edit after Phase 2 if auto-detect fails)

```bash
sudo apt update
sudo apt install -y \
  linux-image-6.8.0-134-generic \
  linux-headers-6.8.0-134-generic \
  linux-modules-6.8.0-134-generic

ls -d /lib/modules/6.8.0-134-generic/build || exit 1
ls -d /usr/src/amdgpu-* || sudo apt install --reinstall -y amdgpu-dkms

AMDGPU_VER=$(basename /usr/src/amdgpu-* | sed 's/amdgpu-//')
sudo dkms install amdgpu/${AMDGPU_VER} -k 6.8.0-134-generic

dkms status | grep 134
ls /lib/modules/6.8.0-134-generic/updates/dkms/amdgpu.ko* || exit 1

echo "DKMS OK — set grub to 6.8.0-134-generic, then reboot"
```

---

## Check driver version (CLI reference)

```bash
rocm-smi --showdriverversion
cat /sys/module/amdgpu/version
modinfo amdgpu | grep ^version
```

DKMS package vs user-facing driver:

| DKMS package | User-facing driver |
|--------------|-------------------|
| `amdgpu/6.16.13-2303411.24.04` | ~30.30.01 |
| `amdgpu/6.16.13-2317211.24.04` | ~30.30.02 |

---

## Related

- RCCL K8s runbook: `network-operator/scripts/k8s-rccl.sh`
- AINIC image tag must match `pds/1.117.5.a.77` on node: `..._ainic-1.117.5-a-77`
- MI355X r17 OOB interface: `enp81s0f1` (not `enp81s0f0` — that is MI325X e13)

---

## BKC artifacts (MI355X-O)

Official BKC bundle for MI350/355H — driver, firmware, and validation packages for this platform:

- **BKC:** H25.17 / RC10
- **Bundle:** `AMD_MI350_355H_01.25.17.10.76`
- **Path:** [dcgpuval-storage — BKC_H25.17 Production & debug official bundle](http://dcgpuval-storage.amd.com/sde-validation/projects%20%28BKC,%20CRD%29/MI355X-O/bkc-artifacts/BKC_H25.17/Production_%26debug_official_bundle/RC10/AMD_MI350_355H_01.25.17.10.76/)

Use this bundle as the source of truth for amdgpu/amdgpu-dkms versions when aligning nodes to **30.30.02** and kernel **6.8.0-134-generic`.
