"""
plot_template.py

그래프 시각화 템플릿
- 모든 설정을 상단 패널에서 관리
- 스위치(on/off) 기능으로 각 옵션 제어

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ============================================================
# ▶▶▶ 사용자 설정 패널 (USER CONFIGURATION PANEL) ◀◀◀
# ============================================================

# -------------------- 스위치 설정 (ON/OFF) --------------------
CONNECT_POINTS = True          # 데이터 포인트를 곡선으로 연결
SHOW_X_VALUES = False          # 각 포인트에 X값 표시
SHOW_Y_VALUES = False           # 각 포인트에 Y값 표시
SHOW_ONLY_MAX = False          # 최대 Y값만 표시 (True면 다른 라벨 무시)
SHOW_HORIZONTAL_LINE = False   # 수평선 표시
SHOW_VERTICAL_LINE = False     # 수직선 표시
SHOW_LEGEND = False             # 범례 표시
SHOW_GRID = True               # 그리드 표시
USE_MANUAL_AXIS_RANGE = True   # 축 범위 수동 설정 사용

# -------------------- 축 설정 --------------------
# X축
X_LABEL = "2DEG Depth (nm)"       # X축 라벨
X_LABEL_SIZE = 22              # X축 라벨 크기
X_TICK_SIZE = 26               # X축 눈금 숫자 크기
X_MIN = 10                      # X축 최소값 (USE_MANUAL_AXIS_RANGE=True일 때)
X_MAX = 100                    # X축 최대값
X_TICKS = [20, 40, 60, 80, 100]  # X축 표기할 값 (빈 리스트면 자동)

# Y축
Y_LABEL = "Optimal Split Gate Gap (nm)"       # Y축 라벨
Y_LABEL_SIZE = 22              # Y축 라벨 크기
Y_TICK_SIZE = 26               # Y축 눈금 숫자 크기
Y_MIN = 10                      # Y축 최소값 (USE_MANUAL_AXIS_RANGE=True일 때)
Y_MAX = 100                     # Y축 최대값
Y_TICKS = [20, 40, 60, 80, 100]  # Y축 표기할 값 (빈 리스트면 자동)

# -------------------- 타이틀 설정 --------------------
GRAPH_TITLE = ""    # 그래프 제목
TITLE_SIZE = 16                # 제목 크기
TITLE_WEIGHT = "bold"          # 제목 굵기 ("normal", "bold")

# -------------------- 데이터 포인트 설정 --------------------
MARKER_STYLE = "o"             # 마커 스타일 ("o", "s", "^", "D", etc.)
MARKER_SIZE = 8                # 마커 크기
LINE_STYLE = "-"               # 선 스타일 ("-", "--", "-.", ":")
LINE_WIDTH = 2.0               # 선 두께
LINE_COLOR = "blue"            # 선/마커 색상

# -------------------- 데이터 라벨 설정 --------------------
LABEL_FONTSIZE = 10            # 라벨 글자 크기
LABEL_FONTWEIGHT = "bold"      # 라벨 굵기
LABEL_OFFSET_X = 0             # 라벨 X 오프셋
LABEL_OFFSET_Y = 1.5           # 라벨 Y 오프셋 (포인트 위로)
LABEL_DECIMALS = 2             # 소수점 자릿수

# -------------------- 보조선 설정 --------------------
H_LINE_Y = 25                  # 수평선 Y 위치
H_LINE_COLOR = "yellow"           # 수평선 색상
H_LINE_STYLE = "--"            # 수평선 스타일
H_LINE_WIDTH = 2               # 수평선 두께
H_LINE_LABEL = "Reference"     # 수평선 범례 라벨

V_LINE_X = 50                  # 수직선 X 위치
V_LINE_COLOR = "green"         # 수직선 색상
V_LINE_STYLE = ":"             # 수직선 스타일
V_LINE_WIDTH = 2               # 수직선 두께
V_LINE_LABEL = "Threshold"     # 수직선 범례 라벨

# -------------------- 범례 설정 --------------------
LEGEND_LOC = "upper right"     # 범례 위치
LEGEND_FONTSIZE = 11           # 범례 글자 크기
DATA_LABEL = "Data"            # 데이터 범례 라벨

# -------------------- Figure 설정 --------------------
FIG_WIDTH = 10                 # Figure 가로 크기
FIG_HEIGHT = 7                 # Figure 세로 크기
DPI = 100                      # 해상도

# -------------------- 데이터 입력 --------------------
# (x, y) 데이터 포인트 직접 입력
X_DATA = [20, 30, 40, 50, 60, 70, 80]
Y_DATA = [27, 38, 50, 60, 74, 84, 96]

# X_DATA = [20, 30, 40, 50, 60, 70, 80]
# Y_DATA = [47.73, 32.19, 24.25, 19.43, 16.19, 13.88, 12.13]

# X_DATA = [27, 38, 50, 60, 74, 84, 96]
# Y_DATA = [47.73, 32.19, 24.25, 19.43, 16.20, 13.88, 12.13]

# ============================================================
# ▶▶▶ 설정 패널 끝 ◀◀◀
# ============================================================


def create_plot():
    """설정값을 기반으로 그래프 생성"""
    
    # Figure 생성
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    
    # 데이터 numpy 배열로 변환
    x = np.array(X_DATA)
    y = np.array(Y_DATA)
    
    # -------------------- 데이터 플롯 --------------------
    if CONNECT_POINTS:
        # 곡선으로 연결
        ax.plot(
            x, y,
            linestyle=LINE_STYLE,
            linewidth=LINE_WIDTH,
            color=LINE_COLOR,
            marker=MARKER_STYLE,
            markersize=MARKER_SIZE,
            label=DATA_LABEL,
        )
    else:
        # 포인트만 표시 (연결 안함)
        ax.scatter(
            x, y,
            s=MARKER_SIZE**2,
            color=LINE_COLOR,
            marker=MARKER_STYLE,
            label=DATA_LABEL,
        )
    
    # -------------------- 데이터 라벨 표시 --------------------
    if SHOW_ONLY_MAX:
        # 최대 Y값만 표시
        max_idx = np.argmax(y)
        label_text = ""
        if SHOW_X_VALUES and SHOW_Y_VALUES:
            label_text = f"({x[max_idx]:.{LABEL_DECIMALS}f}, {y[max_idx]:.{LABEL_DECIMALS}f})"
        elif SHOW_X_VALUES:
            label_text = f"{x[max_idx]:.{LABEL_DECIMALS}f}"
        elif SHOW_Y_VALUES:
            label_text = f"{y[max_idx]:.{LABEL_DECIMALS}f}"
        
        if label_text:
            ax.text(
                x[max_idx] + LABEL_OFFSET_X,
                y[max_idx] + LABEL_OFFSET_Y,
                label_text,
                ha="center",
                va="bottom",
                fontsize=LABEL_FONTSIZE,
                fontweight=LABEL_FONTWEIGHT,
            )
    else:
        # 모든 포인트에 라벨 표시
        if SHOW_X_VALUES or SHOW_Y_VALUES:
            for xi, yi in zip(x, y):
                if SHOW_X_VALUES and SHOW_Y_VALUES:
                    label_text = f"({xi:.{LABEL_DECIMALS}f}, {yi:.{LABEL_DECIMALS}f})"
                elif SHOW_X_VALUES:
                    label_text = f"{xi:.{LABEL_DECIMALS}f}"
                else:  # SHOW_Y_VALUES
                    label_text = f"{yi:.{LABEL_DECIMALS}f}"
                
                ax.text(
                    xi + LABEL_OFFSET_X,
                    yi + LABEL_OFFSET_Y,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=LABEL_FONTSIZE,
                    fontweight=LABEL_FONTWEIGHT,
                )
    
    # -------------------- 보조선 --------------------
    if SHOW_HORIZONTAL_LINE:
        ax.axhline(
            y=H_LINE_Y,
            color=H_LINE_COLOR,
            linestyle=H_LINE_STYLE,
            linewidth=H_LINE_WIDTH,
            label=H_LINE_LABEL,
        )
    
    if SHOW_VERTICAL_LINE:
        ax.axvline(
            x=V_LINE_X,
            color=V_LINE_COLOR,
            linestyle=V_LINE_STYLE,
            linewidth=V_LINE_WIDTH,
            label=V_LINE_LABEL,
        )
    
    # -------------------- 축 설정 --------------------
    # 축 범위
    if USE_MANUAL_AXIS_RANGE:
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
    
    # 축 눈금 수동 지정
    if X_TICKS:
        ax.set_xticks(X_TICKS)
    if Y_TICKS:
        ax.set_yticks(Y_TICKS)
    
    # 축 라벨
    ax.set_xlabel(X_LABEL, fontsize=X_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(Y_LABEL, fontsize=Y_LABEL_SIZE, fontweight="bold")
    
    # 축 눈금 글자 크기
    ax.tick_params(axis='x', labelsize=X_TICK_SIZE)
    ax.tick_params(axis='y', labelsize=Y_TICK_SIZE)
    
    # -------------------- 타이틀 --------------------
    ax.set_title(GRAPH_TITLE, fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT)
    
    # -------------------- 그리드 --------------------
    if SHOW_GRID:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # -------------------- 범례 --------------------
    if SHOW_LEGEND:
        ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    
    return fig, ax


def main():
    """메인 실행 함수"""
    
    print("=" * 60)
    print("그래프 생성 중...")
    print("=" * 60)
    
    # 현재 설정 출력
    print("\n[현재 스위치 설정]")
    print(f"  - 포인트 연결: {CONNECT_POINTS}")
    print(f"  - X값 표시: {SHOW_X_VALUES}")
    print(f"  - Y값 표시: {SHOW_Y_VALUES}")
    print(f"  - 최대값만 표시: {SHOW_ONLY_MAX}")
    print(f"  - 수평선 표시: {SHOW_HORIZONTAL_LINE}")
    print(f"  - 수직선 표시: {SHOW_VERTICAL_LINE}")
    print(f"  - 범례 표시: {SHOW_LEGEND}")
    print(f"  - 그리드 표시: {SHOW_GRID}")
    print(f"  - 축 범위 수동 설정: {USE_MANUAL_AXIS_RANGE}")
    
    print(f"\n[데이터]")
    print(f"  X: {X_DATA}")
    print(f"  Y: {Y_DATA}")
    
    # 그래프 생성
    fig, ax = create_plot()
    
    # 저장
    # output_filename = "output_plot.png"
    # fig.savefig(output_filename, dpi=200, bbox_inches='tight')
    # print(f"\n그래프 저장됨: {output_filename}")
    
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    main()