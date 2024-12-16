from multiprocessing import cpu_count


def _get_num_procs_in_container() -> int:
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as fp:
        cfs_period_us = int(fp.read())

    container_cpus = cfs_quota_us // cfs_period_us

    # For physical machine, the `cfs_quota_us` could be '-1'
    cpus = cpu_count() if container_cpus < 1 else container_cpus
    return cpus


def get_num_procs() -> int:
    try:
        return _get_num_procs_in_container()
    except FileNotFoundError:
        return cpu_count()
