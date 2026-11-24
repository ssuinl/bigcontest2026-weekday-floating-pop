"""Run the approved A3 analysis without changing source files."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MONTHS = [f"2025{month:02d}" for month in range(7, 13)]
COLS = [f"FLOW_POP_CNT_{day}" for day in
        ("MON", "TUS", "WED", "THU", "FRI", "SAT", "SUN")]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
KEYS = ["지역", "STD_YM", "요일"]
METRICS = ["n", "합계", "평균", "중앙값"]
SUM_LABEL = "격자별 요일 일평균 값의 합산 (실제 방문자 수 아님)"


def compare_a2(table, path):
    if not path.exists():
        print(f"[미완료] A2 비교: 파일 없음: {path}")
        return False
    reference = pd.read_csv(path, encoding="utf-8-sig", dtype={"STD_YM": str})
    required = KEYS + METRICS
    if set(reference.columns) != set(required):
        print(f"[미완료] A2 열 구조 확인 필요: {list(reference.columns)}; 예상: {required}")
        return False
    if reference.duplicated(KEYS).any():
        print("[실패] A2 집계표 키 중복")
        return False
    left = table.set_index(KEYS).sort_index()
    right = reference.set_index(KEYS).sort_index()
    if not left.index.equals(right.index):
        print("[실패] A2 집계표 지역/월/요일 키 불일치")
        return False
    passed = True
    for metric in METRICS:
        a, b = left[metric].to_numpy(), right[metric].to_numpy()
        matches = np.isclose(a, b, atol=1e-6, rtol=0, equal_nan=True)
        print(f"[{'통과' if matches.all() else '실패'}] A2 {metric}: 불일치 {int((~matches).sum())}건")
        passed = passed and bool(matches.all())
    return passed


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=root / "2026빅콘테스트_sk통신인구_데이터_레이아웃")
    parser.add_argument("--output-dir", type=Path, default=root / "A3_outputs")
    parser.add_argument("--a2", type=Path, default=root / "A2_지역x월x요일_집계.csv")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    frames = []
    for month in MONTHS:
        frame = pd.read_csv(args.input_dir / f"flow_wkdy_pop_{month}.csv",
                            sep="|", encoding="utf-8-sig",
                            dtype={"BLOCK_CD": str, "STD_YM": str})
        if month == "202512":
            before = len(frame)
            frame = frame.drop_duplicates()
            print(f"CR-001 202512: 처리 전 {before:,}, 처리 후 {len(frame):,}")
            if len(frame) != 121267:
                raise ValueError("202512 처리 후 행 수가 121267과 다릅니다.")
        if set(frame["STD_YM"]) != {month}:
            raise ValueError(f"파일명과 STD_YM 충돌: {month}")
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    # SGIS 행정구역코드 기준 잠정 분류, BLOCK_CD 정의 미확인
    prefixes = source["BLOCK_CD"].str[:5]
    print(f"분석 대상 외 코드 행 수: {int((~prefixes.isin(['11230', '32010'])).sum()):,}")
    aggregate, summary = [], []
    groups = {}
    for region, prefix, bounds in [("강남", "11230", (15369, 15523)),
                                   ("춘천", "32010", (102349, 106630))]:
        for month in MONTHS:
            group = source.loc[(prefixes == prefix) & (source["STD_YM"] == month), COLS].copy()
            groups[region, month] = group
            count = len(group)
            print(f"월별 n {region} {month}: {count:,} [{'통과' if bounds[0] <= count <= bounds[1] else '실패'}]")
            if not bounds[0] <= count <= bounds[1]:
                raise ValueError(f"월별 n 범위 불일치: {region} {month}")
            if group.isna().any().any():
                raise ValueError(f"명세에 결측값 처리 기준 없음: {region} {month}")
            for col, day in zip(COLS, DAYS):
                values = group[col]
                key = dict(zip(KEYS, [region, month, day]))
                aggregate.append({**key, "n": values.count(), "합계": values.sum(),
                                  "평균": values.mean(), "중앙값": values.median()})
                q = values.quantile([.25, .5, .75, .9, .95])
                zeros = int(values.eq(0).sum())
                summary.append({**key, "n": values.count(), "0값 개수": zeros,
                                "0값 비율(%)": zeros / values.count() * 100,
                                "min": values.min(), "Q1": q[.25], "median": q[.5],
                                "Q3": q[.75], "P90": q[.9], "P95": q[.95],
                                "max": values.max(), "mean": values.mean()})
    table, distribution = pd.DataFrame(aggregate), pd.DataFrame(summary)
    if not np.array_equal(table["중앙값"], distribution["median"]):
        raise ValueError("분위수표 median과 A3-1 중앙값 불일치")
    print("[통과] 분위수표 median = A3-1 중앙값")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "A3-1_지역x월x요일_집계.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(args.output_dir / "A3_분포요약.csv", index=False, encoding="utf-8-sig")
    for region in ["강남", "춘천"]:
        for metric in ["합계", "평균", "중앙값"]:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
            label = SUM_LABEL if metric == "합계" else metric
            fig.suptitle(f"{region} 월별 요일 패턴 - {label}", fontsize=15)
            for month, ax in zip(MONTHS, axes.flat):
                rows = table.loc[(table["지역"] == region) & (table["STD_YM"] == month)]
                ax.plot(DAYS, rows[metric], marker="o")
                ax.set_title(f"{month[:4]}-{month[4:]}")
                ax.set_xlabel("요일")
                ax.set_ylabel(label, fontsize=9)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            fig.tight_layout(rect=(0, 0, 1, .95))
            fig.savefig(args.output_dir / f"A3-1_{region}_{metric}.png", dpi=160)
            plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
        fig.suptitle(f"{region} 월별 요일 격자 분포", fontsize=15)
        for month, ax in zip(MONTHS, axes.flat):
            ax.boxplot([groups[region, month][col].to_numpy() for col in COLS])
            ax.set_xticks(range(1, 8), DAYS)
            ax.set_title(f"{month[:4]}-{month[4:]}")
            ax.set_xlabel("요일")
            ax.set_ylabel("격자별 요일 일평균 값")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.tight_layout(rect=(0, 0, 1, .95))
        number = "A3-2" if region == "강남" else "A3-3"
        fig.savefig(args.output_dir / f"{number}_{region}_격자분포.png", dpi=160)
        plt.close(fig)
    complete = compare_a2(table, args.a2)
    print(f"PNG 8개 및 CSV 2개 생성: {args.output_dir}")
    print("검증 완료" if complete else "결과 제출 전 A2 비교 완료 필요")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
