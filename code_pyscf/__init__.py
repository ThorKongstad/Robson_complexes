import re


def parse_mem_to_mb(mem_str):
    """Convert SLURM memory string to MB. Supports K/M/G/T with or without B."""
    m = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?)B?$', mem_str.strip().upper())
    if not m:
        raise ValueError(f"Unrecognised memory format: {mem_str}")
    value, unit = float(m.group(1)), m.group(2)
    return int(value * {'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2}.get(unit, 1))


def parse_script_header(script_path):
    """Read #nprocshared and #mem from script header comments."""
    nprocs, mem_per_cpu = 2, 2
    with open(script_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'#nprocshared=(\d+)', line)
            if m:
                nprocs = int(m.group(1))
            m = re.match(r'#mem=(.+)', line)
            if m:
                mem_per_cpu = parse_mem_to_mb(m.group(1))
    if mem_per_cpu is None:
        raise ValueError("No #mem= line found in script header")
    return nprocs, mem_per_cpu


__all__ = [parse_mem_to_mb, parse_script_header]