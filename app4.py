import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
# Streamlit環境でのMatplotlibの警告・エラー防止
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib import font_manager
from statsmodels.stats.power import TTestIndPower
import math
import random

# --- 日本語フォント対応 ---
try:
    import japanize_matplotlib
except ImportError:
    pass

def set_japanese_font():
    candidates = ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP',
                  'Hiragino Sans', 'Hiragino Maru Gothic Pro', 'MS Gothic', 
                  'Yu Gothic', 'Meiryo']
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            return
set_japanese_font()

# --- Session Stateの初期化 ---
if "df_layout" not in st.session_state:
    st.session_state.df_layout = None
if "treatments" not in st.session_state:
    st.session_state.treatments = []

# --- ページ設定とタイトル ---
st.set_page_config(page_title="第２章 実験計画法", layout="wide")
st.title("第２章　実験計画法「なぜ反復とブロックが必要か」")
st.markdown("解析の前に、**「そもそも正しいデータの取り方をしているか？」** を設計するためのツールです。")

tab1, tab2, tab3 = st.tabs([
    "📏 1. サンプルサイズ計算 (なぜ反復が必要か？)", 
    "🧩 2. ブロックとは？ (ムラを消す設計)", 
    "🗺️ 3. 圃場配置マップ生成 (どう植える？)"
])

# ==========================================
# タブ1: サンプルサイズ計算
# ==========================================
with tab1:
    st.header("📏 サンプルサイズ計算 (検出力分析)")
    st.markdown("新技術（新品種や新肥料など）の効果を証明するために、**最低限いくつ（何株・何区画の反復）のデータが必要か**を統計的に逆算します。")

    st.caption(
        "⚠️ **前提**: この計算は「対照区 vs. 処理区の2群比較（t検定）」の検出力を想定しています。"
        "3群以上の多重比較（Dunnett検定等）では有意水準の補正が入るため、実際に必要な反復数はこれより多くなります。"
        "目安としてご利用ください。"
    )

    st.subheader("🌱 測るデータの単位")
    unit_choice = st.radio(
        "収量などのデータを測る単位はどちらですか？",
        ["株（個体ごとに1つずつ測る）", "区画（1プロットの収量をまとめて測る）"],
        horizontal=True
    )
    unit = "株" if "株" in unit_choice else "区画"

    plants_per_plot = None
    if unit == "区画":
        plants_per_plot = st.number_input(
            "1区画あたりの株数",
            min_value=1, value=10, step=1,
            help="例：10株まとめて1区画として収穫・計量する場合は「10」。株数を増やすほど区画間のばらつき（CV）が小さくなり、必要反復数が減る傾向があります。"
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 過去のデータ（基準）と期待する効果")
        mean1 = st.number_input(f"① 現在の平均値（例：従来品種の1{unit}あたりの収量）", value=500.0, step=10.0)
        std_dev = st.number_input(f"② データのばらつき（標準偏差）", value=50.0, step=5.0,
                                  help=f"過去のデータで、1{unit}ごとの値が平均からどれくらいズレていたかの目安です。")
        mean2 = st.number_input(f"③ 期待する新しい平均値（例：新品種の目標収量）", value=600.0, step=10.0)

        with st.expander("📖 「標準偏差」はどのくらいに設定すればいい？"):
            st.markdown("""
            標準偏差は、「データが平均値から上下にどれくらい散らばっているか」を表す数値です。過去のデータがない場合は、**「変動係数（CV）」**という指標を使って見積もるのが便利です。

            **【変動係数(CV)の求め方】** `変動係数(%) ＝ (標準偏差 ÷ 平均値) × 100`

            つまり、入力する標準偏差に迷ったら、**「 平均値 × 期待するばらつき(%) 」** で逆算して入力してください。農業の野外試験では、このCVが**5%〜15%程度**になるのが一般的です。

            * **CV 5% 程度**（例：平均500なら「25」を入力）: 温室や、極めて管理の行き届いた水田など。
            * **CV 10% 程度**（例：平均500なら「50」を入力）: **標準的な野外圃場試験。** まずはこれを入力するのがおすすめ。
            * **CV 15%〜20%**（例：平均500なら「75〜100」を入力）: ばらつきが大きい状態。土壌ムラが激しい畑や「単株調査」の場合。
            * **CV 20% 以上**: かなりノイズが大きい状態。病害虫の被害や食害があった圃場データなど。
            """)

    with col2:
        st.subheader("🎯 統計的基準（通常は変更不要）")
        alpha = st.slider("有意水準 (α): 誤って「差がある」と判断する確率", 0.01, 0.10, 0.05, 0.01)
        power = st.slider("検出力 (1-β): 本当に差がある時に、正しく見抜ける確率", 0.50, 0.99, 0.80, 0.01)

        with st.expander("📖 統計用語の詳しい解説と目安"):
            st.markdown("""
            * **有意水準 (α)**: 「本当は効果がないのに、たまたま出た偶然の差を『効果がある！』と勘違いしてしまう確率」
              * **目安**: 通常は **0.05 (5%)** に設定します。「間違って使えない新肥料を導入してしまうリスクを5%までは許容する」という意味です。
            * **検出力 (1-β)**: 「本当に効果がある技術をテストした時に、見落とさずに『効果がある』と正しく証明できる確率」
              * **目安**: 通常は **0.80 (80%)** に設定します。これが低いと、画期的な新技術をボツにしてしまう事になります。
            """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(f"必要な{unit}数（反復数）を計算", key="btn_power", type="primary"):
        diff = abs(mean1 - mean2)
        if std_dev <= 0:
            st.error("🚨 標準偏差は0より大きい数値を入力してください。")
        elif diff == 0:
            st.error("🚨 現在の平均と期待する平均に差がありません。")
        else:
            effect_size = diff / std_dev
            analysis = TTestIndPower()
            n_per_group = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, power=power, ratio=1.0)
            n_ceil = math.ceil(n_per_group)

            st.success(f"🎉 **計算結果: 1処理（1群）あたり最低【 {n_ceil} {unit} 】（合計 {n_ceil * 2} {unit}）の反復データが必要です！**")

            if unit == "区画" and plants_per_plot is not None:
                total_plants = n_ceil * 2 * plants_per_plot
                st.info(f"📌 **試験規模の目安**: 1区画 {plants_per_plot} 株 × 合計 {n_ceil * 2} 区画 ＝ **総株数 {total_plants} 株**")

            st.markdown("---")
            is_warning = (unit == "株" and n_ceil >= 10) or (unit == "区画" and n_ceil > 5)

            if not is_warning:
                st.subheader("✨ この反復数なら、十分に実施可能です！")
                st.info(f"""
                1処理あたり **{n_ceil} {unit}（反復）** であれば、通常の野外圃場試験として無理なく設定できる規模です。
                入力いただいた条件であれば、**この少数の反復でも新技術の効果を統計的にしっかりと証明できる可能性が高い**です。このまま上の「タブ2」「タブ3」に進んで、圃場配置マップを作成しましょう！
                """)
            else:
                st.subheader("🤔 この反復数、現実的に準備・調査できますか？")
                if unit == "株" and n_ceil >= 10:
                    st.warning(f"**【警告】1処理あたり {n_ceil} 株の「単株調査」は、多大な労力とノイズを伴います。**")
                    st.markdown("""
                    * ① 1株ごとのラベリングや収穫は煩雑で、サンプルの取り違えリスクが高い。
                    * ② 根元の小さな石や虫食いなど、ミクロな違いが「個体差」という強烈なノイズになる。
                    * ③ 現場で知りたいのは「群落（面積あたり）の収量」であり、単株データとは乖離しやすい。
                    """)
                elif unit == "区画" and n_ceil > 5:
                    st.warning(f"**【警告】1処理あたり {n_ceil} 区画（反復）** という数は、通常の野外圃場試験としては**非常に大規模**であり、場所や労力の確保が困難になる可能性が高いです。")
                    if plants_per_plot is not None and plants_per_plot < 10:
                        st.info(f"💡 **1区画あたりの株数（現在: {plants_per_plot} 株）を増やすことも有効です。** 個体差が平均化されてばらつきが下がり、必要反復数が減る傾向があります。")

                st.markdown("""
                ### 💡 現実的な解決策：複数区画の合算、年次反復、そして高度な解析
                無理に大量の反復を行うのではなく、以下のアプローチが推奨されます。
                1. **「区画」単位での評価**：10株をまとめて1区画とし、個体のノイズを平均化する。
                2. **年次・場所を分ける**：「毎年3反復を3年間」行い、気象変動に対する再現性を評価する。
                3. **一般化線形混合モデル (GLMM) による解析**：年次や場所のムラを「変量効果」として統計モデルに組み込み、純粋な処理の効果だけを高感度で抽出する。
                """)


# ==========================================
# タブ2: ブロックの考え方 (ムラを消す設計)
# ==========================================
with tab2:
    st.header("🧩 ブロックとは？ (場所ムラを消す設計)")
    st.markdown(
        "計算した反復数（区画）を、圃場にどう並べるかが重要です。\n\n"
        "圃場には**肥沃度・日当たり・水はけなどの場所ムラ**があります。"
        "乱塊法（RCBD）では、ブロック内が均質な環境になるように、**ムラの変化方向と直交するように境界線を引き**、帯（ブロック）を作ります。"
        "そして、各ブロック内に全処理を1回ずつ配置することで、場所ムラの影響を排除します。"
    )

    n_trt2 = st.slider("処理数", 2, 6, 3, key="t2_trt")
    n_blk2 = st.slider("ブロック数（＝反復数）", 2, 5, 4, key="t2_blk")
    grad_dir = st.radio("圃場のムラの変化方向", ["左→右に変化（横方向のムラ）", "下→上に変化（縦方向のムラ）"], key="t2_dir", horizontal=True)
    grad_strength = st.slider("ムラの強さ", 1, 5, 3, key="t2_str")

    if "横" in grad_dir:
        n_cols2 = n_blk2
        n_rows2 = n_trt2
    else:
        n_cols2 = n_trt2
        n_rows2 = n_blk2

    base_colors2 = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    trt_colors2 = base_colors2[:n_trt2]

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, max(n_rows2, 3) * 1.0 + 1.0))

    # 左：ブロックなし
    ax_l = axes2[0]
    ax_l.set_xlim(0, n_cols2); ax_l.set_ylim(0, n_rows2)
    ax_l.set_aspect("equal"); ax_l.axis("off")
    # 文字化け防止のため絵文字を削除し、padを25に増やして文字の重なりを解消
    ax_l.set_title("ブロックなし\n（場所ムラと処理の差が混ざる）", fontsize=11, pad=25)

    for y in range(n_rows2):
        for x in range(n_cols2):
            if "横" in grad_dir:
                intensity = x / max(n_cols2 - 1, 1)
            else:
                intensity = y / max(n_rows2 - 1, 1)
            alpha_val = 0.1 + intensity * 0.6 * (grad_strength / 5)
            ax_l.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor="gray", facecolor=(0.85, 0.65, 0.2), alpha=alpha_val, linewidth=0.5))
            ax_l.text(x + 0.5, y + 0.5, "?", ha="center", va="center", fontsize=14, color="#777")

    if "横" in grad_dir:
        ax_l.annotate("", xy=(n_cols2, -0.3), xytext=(0, -0.3), arrowprops=dict(arrowstyle="->", color="saddlebrown", lw=2))
        ax_l.text(n_cols2 / 2, -0.5, "← 肥沃度などが変化 →", ha="center", va="top", fontsize=10, color="saddlebrown", fontweight="bold")
    else:
        ax_l.annotate("", xy=(-0.3, n_rows2), xytext=(-0.3, 0), arrowprops=dict(arrowstyle="->", color="saddlebrown", lw=2))
        ax_l.text(-0.5, n_rows2 / 2, "肥沃度などが変化\n↑", ha="right", va="center", fontsize=10, color="saddlebrown", fontweight="bold")

    # 右：ブロックあり
    ax_r = axes2[1]
    ax_r.set_xlim(0, n_cols2); ax_r.set_ylim(0, n_rows2)
    ax_r.set_aspect("equal"); ax_r.axis("off")
    # 文字化け防止のため絵文字を削除し、padを25に増やして文字の重なりを解消
    ax_r.set_title("ブロックあり（乱塊法）\n（直交するように切り、ブロック内を均一にする）", fontsize=11, pad=25)

    rng2 = random.Random(42)
    for b in range(n_blk2):
        order = list(range(n_trt2))
        rng2.shuffle(order)
        for i, trt_idx in enumerate(order):
            if "横" in grad_dir:
                x = b
                y = i
                intensity = x / max(n_cols2 - 1, 1)
            else:
                x = i
                y = n_rows2 - 1 - b
                intensity = y / max(n_rows2 - 1, 1)
            alpha_bg = 0.1 + intensity * 0.6 * (grad_strength / 5)
            ax_r.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor="none", facecolor=(0.85, 0.65, 0.2), alpha=alpha_bg))
            ax_r.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor="black", facecolor=trt_colors2[trt_idx], alpha=0.72, linewidth=0.8))
            ax_r.text(x + 0.5, y + 0.5, f"処理{trt_idx + 1}", ha="center", va="center", fontsize=10, fontweight="bold")

        if "横" in grad_dir:
            if b < n_blk2 - 1:
                ax_r.plot([b + 1, b + 1], [0, n_rows2], color="navy", linewidth=3.5)
            ax_r.text(b + 0.5, n_rows2 + 0.1, f"ブロック{b + 1}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="navy")
        else:
            if b < n_blk2 - 1:
                y_line = n_rows2 - 1 - b
                ax_r.plot([0, n_cols2], [y_line, y_line], color="navy", linewidth=3.5)
            ax_r.text(-0.1, n_rows2 - 0.5 - b, f"ブロック{b + 1}", ha="right", va="center", fontsize=9, fontweight="bold", color="navy")

    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    with st.expander("📌 ブロック設定のポイント（なぜその向きに切るのか？）", expanded=True):
        if "横" in grad_dir:
            st.markdown("""
            - ムラが**左から右（横方向）**に変化しているため、同じ縦列（Y軸方向）は肥沃度が一定です。
            - そのため、**縦に区切ってブロックを作る（列ブロック）**ことで、各ブロック内は環境が等しくなります。
            """)
        else:
            st.markdown("""
            - ムラが**下から上（縦方向）**に変化しているため、同じ横行（X軸方向）は肥沃度が一定です。
            - そのため、**横に区切ってブロックを作る（行ブロック）**ことで、各ブロック内は環境が等しくなります。
            """)


# ==========================================
# タブ3: 圃場配置マップ生成
# ==========================================
with tab3:
    st.header("🗺️ 圃場配置マップ生成 (どう植える？)")
    st.markdown("場所のムラによる偏りを防ぐため、科学的に妥当な**「ランダム配置マップ」**を作成します。")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("⚙️ 配置の設定")
        design_type = st.radio("実験デザインの選択", [
            "乱塊法 (RCBD): 農業の基本。ブロックごとに全処理を配置",
            "完全無作為化法 (CRD): 均一な環境（温室・インキュベーター）向け",
            "ラテン方格法 (Latin Square): 縦と横、両方のムラを消す高度な配置"
        ])

        t_input = st.text_input("処理名（カンマ区切りで入力）", "無処理, 品種A, 品種B, 品種C")
        treatments = [t.strip() for t in t_input.split(",") if t.strip()]
        n_t = len(treatments)

        if "乱塊法" in design_type:
            st.info("💡 **乱塊法の特徴**: 畑の手前から奥にかけて水はけが変わる場合、同じ水はけのゾーン（ブロック）の中に全処理を1つずつ植えることで、環境のムラを相殺します。")
            blocks = st.number_input("ブロック数（反復数）", min_value=2, value=4, step=1)
            # タブ2の学習内容と連動した設定
            block_dir = st.radio("ブロックの並べ方（タブ2を参照）", [
                "行ブロック (横長に区切る: 縦方向のムラを想定)", 
                "列ブロック (縦長に区切る: 横方向のムラを想定)"
            ])

        elif "完全無作為化法" in design_type:
            st.info("💡 **完全無作為化法の特徴**: 環境のムラが全くない場所で使います。どこに何が来るか完全にランダムなため、屋外の畑で使うと特定の処理が偏って配置される危険があります。")
            reps = st.number_input("1処理あたりの反復数", min_value=2, value=4, step=1)
            cols = st.number_input("表示する圃場の列数", min_value=1, value=n_t, step=1)

        elif "ラテン方格法" in design_type:
            st.info(f"💡 **ラテン方格法の特徴**: 「処理数 = 行数 = 列数」という厳しい制約がありますが、縦方向のムラと横方向のムラを両方とも相殺できる強力な手法です。（現在: {n_t} × {n_t}区画になります）")
            blocks = n_t

        generate_btn = st.button("配置マップを生成", type="primary", use_container_width=True)

        if generate_btn:
            if n_t < 2:
                st.error("🚨 処理を2つ以上入力してください。")
            elif "ラテン方格法" in design_type and n_t < 3:
                st.error("🚨 ラテン方格法は処理数を3つ以上にしてください。")
            else:
                df_layout = pd.DataFrame()
                try:
                    if "乱塊法" in design_type:
                        layout = []
                        for i in range(int(blocks)):
                            t = treatments.copy()
                            np.random.shuffle(t)
                            layout.append(t)
                        
                        if "行ブロック" in block_dir:
                            df_layout = pd.DataFrame(layout,
                                                     index=[f"ブロック{i+1}" for i in range(int(blocks))],
                                                     columns=[f"区画{i+1}" for i in range(n_t)])
                        else:
                            # 列ブロックの場合は転置する
                            df_layout = pd.DataFrame(layout).T
                            df_layout.columns = [f"ブロック{i+1}" for i in range(int(blocks))]
                            df_layout.index = [f"区画{i+1}" for i in range(n_t)]

                    elif "完全無作為化法" in design_type:
                        total = n_t * int(reps)
                        rows_n = math.ceil(total / int(cols))
                        all_plots = treatments * int(reps)
                        while len(all_plots) < rows_n * int(cols):
                            all_plots.append("空き")
                        np.random.shuffle(all_plots)
                        df_layout = pd.DataFrame(np.array(all_plots).reshape((rows_n, int(cols))))
                        df_layout.index = [f"行{i+1}" for i in range(rows_n)]
                        df_layout.columns = [f"列{i+1}" for i in range(int(cols))]

                    elif "ラテン方格法" in design_type:
                        base = np.arange(n_t)
                        square = np.array([np.roll(base, -i) for i in range(n_t)])
                        np.random.shuffle(square)
                        square = square[:, np.random.permutation(n_t)]
                        t_perm = np.random.permutation(n_t)
                        df_layout = pd.DataFrame(
                            [[treatments[t_perm[val]] for val in row] for row in square],
                            index=[f"行{i+1}" for i in range(n_t)],
                            columns=[f"列{i+1}" for i in range(n_t)]
                        )

                    st.session_state.df_layout = df_layout
                    st.session_state.treatments = treatments

                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

    with c2:
        if st.session_state.df_layout is not None:
            st.subheader("📍 生成された圃場配置マップ")
            df_layout = st.session_state.df_layout
            saved_treatments = st.session_state.treatments
            n_t_saved = len(saved_treatments)

            t_dict = {t: i for i, t in enumerate(saved_treatments)}
            t_dict["空き"] = -1
            df_num = df_layout.replace(t_dict).astype(float)

            base_colors = sns.color_palette("pastel", n_t_saved)
            has_empty = "空き" in df_layout.to_numpy()
            if has_empty:
                all_colors = [(0.9, 0.9, 0.9)] + list(base_colors)
                val_min = -1
            else:
                all_colors = list(base_colors)
                val_min = 0

            cmap = mcolors.ListedColormap(all_colors)
            bounds = np.arange(val_min - 0.5, n_t_saved + 0.5, 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(figsize=(8, df_num.shape[0] * 1.0 + 1))
            sns.heatmap(df_num, annot=df_layout, fmt="", ax=ax,
                        cmap=cmap, norm=norm, cbar=False,
                        linewidths=2, linecolor='white',
                        annot_kws={"size": 14, "weight": "bold"})
            plt.yticks(rotation=0)
            ax.xaxis.tick_top()
            st.pyplot(fig)

            csv = df_layout.to_csv().encode('utf-8-sig')
            st.download_button("📥 この配置図をCSVでダウンロード", csv, "field_layout.csv", "text/csv", use_container_width=True)
