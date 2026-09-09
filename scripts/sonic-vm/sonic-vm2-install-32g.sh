#!/bin/bash
# Phase 2: HD boot + HTTP server for onie-nos-install (after ONIE embed).
# Usage: sudo bash sonic-vm2-install-32g.sh
set -euo pipefail

VM="${VM:-sonic-vm2}"
BIN="${BIN:-/etc/aifm/sonic-vm/SONIC-128/sonic-broadcom.bin}"
DISK="${DISK:-/var/lib/libvirt/images/sonic/${VM}-disk.qcow2}"
HOST_IP="${HOST_IP:-192.168.122.1}"
HTTP_PORT="${HTTP_PORT:-8081}"

virsh destroy "$VM" 2>/dev/null || true
sleep 2
virsh undefine "$VM" 2>/dev/null || true

cat > /tmp/sonic-vm-install.xml <<XML
<domain type='kvm'>
  <name>${VM}</name>
  <memory unit='KiB'>8388608</memory>
  <vcpu placement='static'>4</vcpu>
  <os>
    <type arch='x86_64' machine='pc'>hvm</type>
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
      <source file='${BIN}'/>
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

virsh define /tmp/sonic-vm-install.xml
virsh start "$VM"

BIN_DIR="$(dirname "$BIN")"
BIN_NAME="$(basename "$BIN")"
pkill -f "python3 -m http.server ${HTTP_PORT}" 2>/dev/null || true
cd "$BIN_DIR"
nohup python3 -m http.server "${HTTP_PORT}" --bind "${HOST_IP}" >/tmp/${VM}-http.log 2>&1 &
sleep 1
curl -sfI "http://${HOST_IP}:${HTTP_PORT}/${BIN_NAME}" | head -2

echo "VM started. Console: sudo virsh console ${VM}"
echo "In ONIE:"
echo "  ip link set eth0 up"
echo "  udhcpc -i eth0"
echo "  onie-nos-install http://${HOST_IP}:${HTTP_PORT}/${BIN_NAME}"
