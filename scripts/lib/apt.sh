#!/usr/bin/env bash

apt_get_update() {
  local attempts="${APT_GET_UPDATE_ATTEMPTS:-3}"
  local delay_sec="${APT_GET_UPDATE_DELAY_SEC:-5}"
  local i

  for i in $(seq 1 "${attempts}"); do
    if DEBIAN_FRONTEND=noninteractive apt-get update; then
      return 0
    fi

    if [ "${i}" -lt "${attempts}" ]; then
      echo "apt-get update failed (attempt ${i}/${attempts}); retrying in ${delay_sec}s..."
      sleep "${delay_sec}"
    fi
  done

  echo "apt-get update failed after ${attempts} attempts." >&2
  return 1
}
