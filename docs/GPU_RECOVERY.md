# Dual-V100 PCIe/NVLink recovery gate

The inference profiles in this repository require both cards to be usable
before ComfyUI is launched.  A driver package being installed is not enough.

## Observed host fault

This host initially validated correctly with two Tesla V100-SXM2 cards, NV6
topology, six active NVLinks per GPU, CUDA P2P, and NCCL.  It later logged:

```text
Xid 79: GPU has fallen off the bus
Xid 154: GPU Reset Required
```

The shared PLX fabric also reported PCIe AER physical-layer receive errors,
fatal SDES, a downstream link reset, and failed recovery.  Subsequent Xid 74
messages are failed NVLink retraining attempts; they are not evidence that
the NVIDIA driver package is mismatched.

## Safe recovery sequence

1. Stop any `watch nvidia-smi` or other loop querying NVLink.  It generates
   repeated retraining errors while the cards are already unavailable.
2. Preserve a diagnostics report before rebooting:

   ```bash
   cd /tmp
   sudo nvidia-bug-report.sh
   ```

3. Perform a clean host reboot:

   ```bash
   sudo systemctl reboot
   ```

4. Before starting ComfyUI, require all checks below to pass:

   ```bash
   nvidia-smi
   nvidia-smi topo -m
   nvidia-smi nvlink --status
   nvidia-smi topo -p2p r
   nvidia-smi topo -p2p w
   nvidia-smi topo -p2p a
   INSTALL_ROOT=$HOME/minimax-h3 ./scripts/check_nvlink.sh
   ```

Healthy output includes two V100s, `NV6` between GPU0 and GPU1, six active
links on each GPU, and `OK` for read/write/atomic P2P checks.

Because the original fault followed an initially healthy snapshot, run a
ten-minute transfer soak before trusting the host with a long generation:

```bash
NVLINK_SOAK_SECONDS=600 INSTALL_ROOT=$HOME/minimax-h3 ./scripts/soak_nvlink_v100.sh
```

The soak runs the one-shot topology/P2P/NCCL checks first, then performs
correctness-checked full-duplex CUDA peer copies continuously without polling
`nvidia-smi` in a loop.  A failure is evidence of a hardware/fabric stability
problem, not a model or ComfyUI configuration error.

Do not make a kernel-module reload the primary recovery path after Xid 79.
The PCIe AER subsystem has already failed to recover the downstream links, so
a complete PCIe/PLX/GPU reset through reboot is more reliable.  If a reboot
does not restore both cards, fully power-cycle the host and investigate the
PLX/PCIe path, SXM carrier/NVLink fabric, power delivery, and seating before
reinstalling the driver.
