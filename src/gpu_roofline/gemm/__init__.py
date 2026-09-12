from .api import DeviceSpec, KernelConfig, predict_configs, autotune, run
from .tiles import TileLayout, DeviceSpec_from_probe, is_feasible, enumerate_candidate_configs

def _smoke_test_api() -> None:
    """Verify the API stubs are type-correct (raise NotImplementedError, not TypeError)."""
    import torch
    # DeviceSpec (stub — from_probe not yet implemented)
    dev = DeviceSpec(name="stub", peak_fp32_gflops=8100, peak_bw_gbps=320,
                     ridge_flop_per_byte=25.3, smem_per_sm_bytes=49152,
                     regs_per_sm=65536, max_threads_sm=1024, cc=(7, 5))
    try:
        predict_configs(dev, 2048, 2048, 2048)
    except NotImplementedError:
        pass
    try:
        autotune(dev, 2048, 2048, 2048)
    except NotImplementedError:
        pass
    try:
        run(torch.zeros(4,4,device="cuda"), torch.zeros(4,4,device="cuda"), dev=dev)
    except NotImplementedError:
        pass
    # KernelConfig properties should work without implementation
    cfg = KernelConfig(BM=128, BN=128, BK=32, WM=4, WN=4, num_stages=2)
    assert cfg.smem_bytes == (128*32 + 32*128) * 4
    assert cfg.threads_per_block == (128//4) * (128//4)
    assert abs(cfg.arithmetic_intensity - 32.0) < 1e-6
    print("[ok] API stubs type-correct; KernelConfig properties correct")
    print(f"     KernelConfig(128,128,32,4,4,2): AI={cfg.arithmetic_intensity:.0f} FLOP/byte, "
          f"smem={cfg.smem_bytes//1024} KB, threads={cfg.threads_per_block}")

if __name__ == "__main__":
    _smoke_test_api()
