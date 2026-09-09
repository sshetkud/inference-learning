#!/bin/bash
# Phase 1: blank 32G NVMe disk + boot ONIE recovery ISO for embed.
# Usage: sudo VM=sonic-vm2 bash recreate_sonic_vm2_32g.sh
set -euo pipefail

VM="${VM:-sonic-vm2}"
DISK="${DISK:-/var/lib/libvirt/images/sonic/${VM}-disk.qcow2}"
ONIE_ISO="${ONIE_ISO:-/etc/aifm/sonic-vm/onie-recovery-x86_64-amd_anacapa-r0_smallerdisk.iso}"
DISK_SIZE="${DISK_SIZE:-32G}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

virsh destroy "$VM" 2>/dev/null || true
virsh undefine "$VM" 2>/dev/null || true

rm -f "$DISK"
qemu-img create -f qcow2 "$DISK" "$DISK_SIZE"
chown libvirt-qemu:kvm "$DISK"
chmod 660 "$DISK"
ls -lh "$DISK" "$ONIE_ISO"

cat > /tmp/sonic-vm-embed.xml <<XML
<domain type='kvm'>
  <name>${VM}</name>
  <memory unit='KiB'>8388608</memory>
  <vcpu placement='static'>4</vcpu>
  <os>
    <type arch='x86_64' machine='pc'>hvm</type>
    <boot dev='cdrom'/>
    <boot dev='hd'/>
  </os>
  <features><acpi/><apic/></features>
  <cpu mode='host-passthrough'/>
  <clock offset='utc'/>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>restart</on_reboot>
  <on_crash>destroy</on_crash>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='cdrom'>
      <driver name='qemu' type='raw'/>
      <source file='${ONIE_ISO}'/>
      <target dev='hdc' bus='ide'/>
      <readonly/>
    </disk>
    <interface type='network'>
      <source network='default'/>
      <model type='e1000'/>
    </interface>
    <serial type='pty'><target port='0'/></serial>
    <console type='pty'><target type='serial' port='0'/></console>
    <input type='tablet' bus='usb'/>
    <memballoon model='none'/>
  </devices>
  <qemu:commandline xmlns:qemu='http://libvirt.org/schemas/domain/qemu/1.0'>
    <qemu:arg value='-drive'/>
    <qemu:arg value='file=${DISK},format=qcow2,if=none,id=NVME0'/>
    <qemu:arg value='-device'/>
    <qemu:arg value='nvme,drive=NVME0,serial=nvme-0'/>
  </qemu:commandline>
</domain>
XML

virsh define /tmp/sonic-vm-embed.xml
virsh start "$VM"
virsh list --all
echo "Embed ONIE to /dev/nvme0n1: sudo virsh console ${VM}"
echo "After embed success: sudo bash ${SCRIPT_DIR}/sonic-vm2-install-32g.sh"
