import sys
import pandas as pd
import numpy as np
import json
import os.path
import math

print(
    "\n\t\textractDataFromJsons.py: ATTN: only many_gemms.sh should call this script."
)


def checkCorrectness(df):
    dma_cc_e2e_diffs = []
    for i in range(0, 8):
        # assert that global sim and compute complex end-to-end time is the same for each core
        assert (df[f"Global Sim E2E_cc_{i}"] == df[f"Core Complex E2E_cc_{i}"]).all()
        # assert the sum of all regions == sum of overlap stall + compute + prologue + epilogue
        assert (
            df[f"Sum Region Cycles_cc_{i}"] == df[f"Sum Compute+Stall+Pro+Epi_cc_{i}"]
        ).all()
        dma_cc_e2e_diffs.append(
            (abs(df["Global Sim E2E_dma"] - df[f"Core Complex E2E_cc_{i}"])).max()
        )
      
    print("\t\t",end='')
    print(f"max e2e dma vs cc diff time is {max(dma_cc_e2e_diffs)}: {dma_cc_e2e_diffs}")

    # dma E2E global time and each compute core's E2E global time should differ by an acceptably small amount
    threshold = 30
    assert max(dma_cc_e2e_diffs) < threshold  # what is an acceptable threshold??
   
    return True


# Every trace file contains a group of timed regions
# For each compute core's trace, we extract information by
# taking the sum over a set/subset of the timed regions
def parseComputeCoreTrace(core_json, rgc, idx):
    d = {}
    # Global simulation end to end time for the compute core
    tstart = core_json[0]["tstart"]
    tend = core_json[rgc - 1]["tend"]
    d[f"Global Sim E2E_cc_{idx}"] = tend - tstart + 1
    # Compute Complex end to end time for the compute core
    start = core_json[0]["start"]
    end = max(core_json[0]["end_fpss"], core_json[rgc - 1]["end"])
    d[f"Core Complex E2E_cc_{idx}"] = end - start + 1
    # Sum of "cycles" over all trace regions
    sum = 0
    for region in core_json:
        sum = sum + region["cycles"]
    d[f"Sum Region Cycles_cc_{idx}"] = sum

    # Double buffering epilogue and prologue
    prologueTime = core_json[0]["cycles"]
    epilogueTime = core_json[-1]["cycles"]
    d[f"Before Computation_cc_{idx}"] = prologueTime
    d[f"After Computation_cc_{idx}"] = epilogueTime
    # legacy "Kernel Time" calculation
    coreComplexStart = core_json[1]["start"]  # start time of first compute region
    end = core_json[rgc - 2]["end"]  # coreComplex end
    end_fpss = core_json[rgc - 2]["end_fpss"]  # end of last floating point instruction
    latestEnd = max(end, end_fpss)
    cycles = latestEnd - coreComplexStart + 1
    # structure of trace
    # region 0 is the proglogue
    # region 1 is the SSR Config for the first compute tile
    # region 2 the first compute tile computation
    # region 3 is the time spent after the first compute tile and before the next one
    d[f"core{idx}"] = int(cycles)
    # Overlap Stall Time: time waiting for next tile to copy in, results to write out, or for other compute cores to finish
    # we want every other third region starting from region 3 until the epilogue
    overlapStallTime = 0
    for i in range(3, rgc - 1, 3):
        overlapStallTime = overlapStallTime + core_json[i]["cycles"]
    # Raw Compute Time: cumulative time spent processing a tile
    # we want every third region starting from region 2 until the epilogue
    check = 0
    rawComputeTime = 0
    for i in range(2, rgc - 1, 3):
        rawComputeTime = rawComputeTime + core_json[i]["cycles"]
        check = check + 1
    cc_tiles = (rgc - 1) / 3
    # reality check
    if check != cc_tiles:
        raise Exception(
            f"Error: Number of raw compute regions ({check}) is not the same as compute core tile count ({cc_tiles})!"
        )
    # SSR Config Time
    # we want every third region starting from region 1 until the epilogue
    ssrConfigTime = 0
    for i in range(1, rgc - 1, 3):
        ssrConfigTime = ssrConfigTime + core_json[i]["cycles"]
    d[f"SSR Config Time_cc_{idx}"] = ssrConfigTime
    d[f"Overlap Stall Time_cc_{idx}"] = overlapStallTime
    d[f"Raw Compute Time_cc_{idx}"] = rawComputeTime
    d[f"Sum Raw Compute + Overlap Stall_cc_{idx}"] = rawComputeTime + overlapStallTime
    d[f"Sum Compute+Stall+Pro+Epi_cc_{idx}"] = (
        rawComputeTime + overlapStallTime + prologueTime + epilogueTime + ssrConfigTime
    )
    d[f"Sum Compute + SSR Configs_cc_{idx}"] = (
        rawComputeTime + ssrConfigTime
    )
    d[f"cc_tiles_cc_{idx}"] = (rgc - 1) / 3
    return d


def regionCount(M, N, K, m, n, k, idx):
    m_cluster_tiles = int(M / m) * math.ceil(N / n) * math.ceil(K / k) # cc tiles with cluster tile dim m
    m_rem_cluster_tiles = 1 * math.ceil(N / n) * math.ceil(K / k)      # cc tiles with cluster tile dim m_rem
    rem_m = M % m
    if idx < rem_m: # both rem_m and m cluster tiles
        cc_tiles = m_cluster_tiles + m_rem_cluster_tiles
    else: # only m cluster tiles
        cc_tiles = m_cluster_tiles
    rc = 3 * cc_tiles + 1 # 1 for the prologue region, 3 for [ssr_config, compute, and stall regions] per cc tile.
    return rc


def main():
    if len(sys.argv) != 9:
        print("\t", end="")
        print(
            f"USAGE: Requires two string arguments, experiment name and the full path to the experiment's logs folder, followed by M N K m n k.\nYou passed in {len(sys.argv)} args"
        )
    else:
        expName = sys.argv[1]
        logs = sys.argv[2]
        if not os.path.exists(logs):
            print("\t\t", end="")
            print(
                f"extractKernelTimeFromJsons.py: Error: directory {logs} does not exist."
            )
            return 1
        M = int(sys.argv[3])
        N = int(sys.argv[4])
        K = int(sys.argv[5])
        m = int(sys.argv[6])
        n = int(sys.argv[7])
        k = int(sys.argv[8])

        # trace file names are hardcoded
        dmaFileName = f"{logs}/hart-trace_hart_00008-perf.json"
        computeCoreFileNames = [
            f"{logs}/hart-trace_hart_00000-perf.json",
            f"{logs}/hart-trace_hart_00001-perf.json",
            f"{logs}/hart-trace_hart_00002-perf.json",
            f"{logs}/hart-trace_hart_00003-perf.json",
            f"{logs}/hart-trace_hart_00004-perf.json",
            f"{logs}/hart-trace_hart_00005-perf.json",
            f"{logs}/hart-trace_hart_00006-perf.json",
            f"{logs}/hart-trace_hart_00007-perf.json",
        ]
        computeCores = []
        missing_compute_jsons = False

        # region count reality check — skip cores whose JSON wasn't generated (TRACE_DMA_ONLY mode)
        for idx in range(0, len(computeCoreFileNames)):
            f = computeCoreFileNames[idx]
            if not os.path.exists(f):
                missing_compute_jsons = True
                continue
            rgc = regionCount(M, N, K, m, n, k, idx)
            with open(f) as json_file:
                data = json.load(json_file)
                if rgc != len(data):
                    raise Exception(
                        f"PARSE ERROR! JSON {f} contains an an unexpected number of regions: Expected:{rgc} Actual:{len(data)}"
                    )
                else:
                    computeCores.append((f, rgc, idx))
        tracesInfo = {}

        # extract info from compute core traces; fall back to analytics when JSON is absent
        coreComplexStarts = []
        coreComplexEnds = []
        for idx in range(0, len(computeCoreFileNames)):
            f = computeCoreFileNames[idx]
            if not os.path.exists(f):
                # compute cc_tiles analytically; timing values unavailable
                rgc = regionCount(M, N, K, m, n, k, idx)
                cc_tiles = (rgc - 1) / 3
                tracesInfo[f"cc_tiles_cc_{idx}"] = cc_tiles
                for col in [
                    f"Global Sim E2E_cc_{idx}", f"Core Complex E2E_cc_{idx}",
                    f"Sum Region Cycles_cc_{idx}", f"Before Computation_cc_{idx}",
                    f"After Computation_cc_{idx}", f"core{idx}",
                    f"SSR Config Time_cc_{idx}", f"Overlap Stall Time_cc_{idx}",
                    f"Raw Compute Time_cc_{idx}", f"Sum Raw Compute + Overlap Stall_cc_{idx}",
                    f"Sum Compute+Stall+Pro+Epi_cc_{idx}", f"Sum Compute + SSR Configs_cc_{idx}",
                ]:
                    tracesInfo[col] = -1
            else:
                rgc = regionCount(M, N, K, m, n, k, idx)
                with open(f) as json_file:
                    data = json.load(json_file)
                    cc_trace_info = parseComputeCoreTrace(data, rgc, idx)
                    tracesInfo.update(cc_trace_info)
                    coreComplexStarts.append(data[1]["start"])
                    coreComplexEnds.append(data[rgc - 2]["end"])
        if coreComplexStarts:
            tracesInfo["Kernel Time"] = max(coreComplexEnds) - min(coreComplexStarts) + 1
        else:
            tracesInfo["Kernel Time"] = -1

        # extract info from dma core trace
        with open(dmaFileName) as json_file:
            data = json.load(json_file)
            # region count reality check
            if len(data) != 1:
                raise Exception(
                    f"ERROR! Expected DMA trace to contain 1 region but it contains {len(data)}"
                )
            dma_cycles = data[0]["cycles"]
            tracesInfo["dma"] = dma_cycles
            # Global simulation end to end time for the dma core
            tstart = data[0]["tstart"]
            tend = data[0]["tend"]
            cycle_q = data[0]["end"]
            tracesInfo["Global Sim E2E_dma"] = tend - tstart + 1
            tracesInfo["dma cycle_q"] = cycle_q

        # convert dictionary of traces info to data frame
        row = list(tracesInfo.values())
        cols = list(tracesInfo.keys())
        timeData = np.array(row, dtype="int").reshape(1, len(row))
        df = pd.DataFrame(data=timeData, columns=cols)

        # add more information, some of them strings
        def sumOverComputeCores(df, cat):
            return (
                df[f"{cat}_cc_0"]
                + df[f"{cat}_cc_1"]
                + df[f"{cat}_cc_2"]
                + df[f"{cat}_cc_3"]
                + df[f"{cat}_cc_4"]
                + df[f"{cat}_cc_5"]
                + df[f"{cat}_cc_6"]
                + df[f"{cat}_cc_7"]
            )

        df["Total CC Tiles"] = sumOverComputeCores(df, "cc_tiles")
        df["Overlap Stall Time Total"] = sumOverComputeCores(df, "Overlap Stall Time")
        df["Raw Compute Time Total"] = sumOverComputeCores(df, "Raw Compute Time")
        df["SSR Config Time Total"] = sumOverComputeCores(df, "SSR Config Time")
        df["Sum Compute + SSR Configs Total"] = sumOverComputeCores(df,"Sum Compute + SSR Configs")
        df["FakeNN JSON Name"] = expName
        for i in range(0, 8):
            if missing_compute_jsons:
                df[f"Diff from dma E2E_cc_{i}"] = -1
                df[f"Sum Regions - Core Complex E2E_cc_{i}"] = -1
            else:
                df[f"Diff from dma E2E_cc_{i}"] = abs(
                    df["Global Sim E2E_dma"] - df[f"Core Complex E2E_cc_{i}"]
                )
                df[f"Sum Regions - Core Complex E2E_cc_{i}"] = abs(
                    df["Global Sim E2E_dma"] - df[f"Core Complex E2E_cc_{i}"]
                )

        # check for glaring errors (skip when compute core traces were not available)
        if not missing_compute_jsons and not checkCorrectness(df):
            return 1
        # export results to csv
        df.to_csv(f"{logs}/{expName}.csv", index=False)
        return 0


if __name__ == "__main__":
    if main() != 0:
        raise Exception("")
