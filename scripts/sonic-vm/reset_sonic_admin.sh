#!/bin/bash
# Reset admin password on offline SONiC VM disk (default: YourPaSsWoRd).
# Usage: sudo VM=sonic-vm2 PASS='YourPaSsWoRd' bash reset_sonic_admin.sh
set -euo pipefail

VM="${VM:-sonic-vm2}"
DISK="${DISK:-/var/lib/libvirt/images/sonic/${VM}-disk.qcow2}"
PASS="${PASS:-YourPaSsWoRd}"
MNT="${MNT:-/mnt/${VM}}"
NBD="${NBD:-/dev/nbd0}"
SQ_TMP="${SQ_TMP:-/tmp/sonic-sq-shadow}"

virsh destroy "$VM" 2>/dev/null || true
sleep 2

modprobe nbd max_part=16
qemu-nbd --disconnect "$NBD" 2>/dev/null || true
qemu-nbd --connect="$NBD" "$DISK"
sleep 2
partprobe "$NBD"

SONIC_PART="${NBD}p3"
if [[ ! -b "$SONIC_PART" ]]; then
  echo "ERROR: ${SONIC_PART} not found. lsblk:" >&2
  lsblk "$NBD" >&2
  exit 1
fi

mkdir -p "$MNT"
mount "$SONIC_PART" "$MNT"

IMAGE_DIR="$(ls -d "${MNT}"/image-* 2>/dev/null | head -1)"
if [[ -z "$IMAGE_DIR" ]]; then
  echo "ERROR: no image-* dir under ${MNT}" >&2
  exit 1
fi

RW_ETC="${IMAGE_DIR}/rw/etc"
mkdir -p "$RW_ETC"

rm -rf "$SQ_TMP"
unsquashfs -f -d "$SQ_TMP" "${IMAGE_DIR}/fs.squashfs" etc/shadow

HASH="$(openssl passwd -6 "$PASS")"
cp "${SQ_TMP}/etc/shadow" "${RW_ETC}/shadow"
sed -i "s|^admin:.*|admin:${HASH}:20656:0:99999:7:::|" "${RW_ETC}/shadow"
chmod 640 "${RW_ETC}/shadow"
chown root:shadow "${RW_ETC}/shadow"
grep '^admin:' "${RW_ETC}/shadow"

umount "$MNT" || true
qemu-nbd --disconnect "$NBD" || true
rm -rf "$SQ_TMP"

virsh start "$VM"
virsh list --all
echo "Login: admin / ${PASS}"
