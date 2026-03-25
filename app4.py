import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import japanize_matplotlib
from statsmodels.stats.power import TTestIndPower
import math

# --- Session Stateの初期化 ---
if "df_layout" not in st.session_state:
    st.session_state.df_layout = None
if "treatments" not in st.session_state:
    st.session_state.treatments = []

st.set_page_config(page_title="実験計画アプリ", layout="wide")
st.title("🚜 農業データ・実験計画＆サンプルサイズ計算機")
st.markdown("解析の前に、**「そもそも正しいデータの取り方をしているか？」** を設計するためのツールです。")

tab1, tab2 = st.tabs(["📏 1. サンプルサイズ計算 (何株・何区画必要？)", "🗺️ 2. 圃場配置マップ生成 (どう植える？)"])

# ==========================================
# タブ1: サンプルサイズ計算
# ==========================================
with tab1:
    st.header("📏 サンプルサイズ計算 (検出力分析)")
    st.markdown("新技術（新品種や新肥料など）の効果を証明するために、**最低限いくつ（何株・何区画）のデータが必要か**を統計的に逆算します。")

    # ★追加: 但し書き
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
            標準偏差は、「データが平均値から上下にどれくらい散らばっているか」を表す数値です。過去のデータがない場合は、**「変動係数（CV: Coefficient of Variation）」**という指標を使って見積もるのが便利です。

            **【変動係数(CV)の求め方】** `変動係数(%) ＝ (標準偏差 ÷ 平均値) × 100`

            つまり、入力する標準偏差に迷ったら、**「 平均値 × 期待するばらつき(%) 」** で逆算して入力してください。農業の野外試験では、このCVが**5%〜15%程度**になるのが一般的です。

            * **CV 5% 程度**（例：平均500なら「25」を入力）: 非常に均一。環境が制御された温室や、極めて管理の行き届いた水田など。
            * **CV 10% 程度**（例：平均500なら「50」を入力）: **標準的な野外圃場試験。** まずはこの数値を入力してみるのがおすすめです。
            * **CV 15%〜20%**（例：平均500なら「75〜100」を入力）: ばらつきが大きい状態。土壌のムラが激しい畑、または「1区画」ではなく「単株」で調査した場合などに発生しやすくなります。
            * **CV 20% 以上**: かなりノイズが大きい状態。局所的な病害虫の被害や、動物の食害などトラブルがあった圃場のデータです。
            """)

    with col2:
        st.subheader("🎯 統計的基準（通常は変更不要）")
        alpha = st.slider("有意水準 (α): 誤って「差がある」と判断する確率", 0.01, 0.10, 0.05, 0.01)
        power = st.slider("検出力 (1-β): 本当に差がある時に、正しく見抜ける確率", 0.50, 0.99, 0.80, 0.01)

        with st.expander("📖 統計用語の詳しい解説と目安"):
            st.markdown("""
            * **有意水準 (α)**: 「本当は効果がないのに、たまたま出た偶然の差を『効果がある！』と勘違いしてしまう確率（第一種の過誤）」です。
              * **目安**: 通常は **0.05 (5%)** に設定します。農業現場の言葉で言えば、「間違って『使えない新肥料』を導入してしまうリスクを5%までは許容する」という意味になります。
            * **検出力 (1-β)**: 「本当に効果がある優れた技術をテストした時に、その効果を見落とさずに『効果がある』と正しく証明できる確率」です。
              * **目安**: 通常は **0.80 (80%)** に設定します。これが低いと、せっかくの「画期的な新技術」を見逃してボツにしてしまう（第二種の過誤）という、非常にもったいないことになります。
            """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(f"必要な{unit}数を計算", key="btn_power", type="primary"):
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

            st.success(f"🎉 **計算結果: 1処理（1群）あたり最低【 {n_ceil} {unit} 】（合計 {n_ceil * 2} {unit}）のデータが必要です！**")

            if unit == "区画" and plants_per_plot is not None:
                total_plants = n_ceil * 2 * plants_per_plot
                st.info(f"📌 **試験規模の目安**: 1区画 {plants_per_plot} 株 × 合計 {n_ceil * 2} 区画 ＝ **総株数 {total_plants} 株**")

            st.markdown("---")
            is_warning = (unit == "株" and n_ceil >= 10) or (unit == "区画" and n_ceil > 5)

            if not is_warning:
                st.subheader("✨ この反復数なら、十分に実施可能です！")
                st.info(f"""
                1処理あたり **{n_ceil} {unit}（反復）** であれば、通常の野外圃場試験として無理なく設定できる規模です。

                入力いただいた条件（期待する差が十分に大きい、またはばらつきが小さく抑えられている）であれば、**この少数の反復でも新技術の効果を統計的にしっかりと証明できる可能性が高い**です。このまま上の「タブ2」に進んで、圃場配置マップを作成しましょう！
                """)

                with st.expander("📝 現場でのワンポイントアドバイス"):
                    st.markdown("""
                    * 野外試験では、局所的な虫害や水たまりなど、想定外のトラブルでデータが欠損することがあります。もし畑のスペースに余裕があれば、**計算結果より1〜2反復ほど多めに（予備として）植えておく**とより安心です。
                    * 万が一、今年の試験で予期せぬノイズが入り有意差が出なかった場合でも、来年も試験を継続して「年次反復」とすることで、効果を証明できる道が残されています。
                    """)
            else:
                st.subheader("🤔 この反復数、現実的に準備・調査できますか？")

                if unit == "株" and n_ceil >= 10:
                    st.warning(f"""
                    **【警告】1処理あたり {n_ceil} 株（合計 {n_ceil * 2} 株）の「単株調査」は、多大な労力とノイズを伴います。** 場所としては確保できても、1株ごとに分けてデータを取る（収穫・脱穀・計量など）ことには以下の**強いデメリット**があります。
                    """)
                    st.markdown("""
                    * **① 膨大な労力とヒューマンエラー:** 1株ごとに別々の袋に入れ、ラベリングし、混ざらないように処理・記録する作業は非常に煩雑で、サンプルの取り違えリスクが跳ね上がります。
                    * **② 「個体差」という強烈なノイズ:** 根元の小さな石、わずかな虫食い、隣の株との競合など、ミクロな違いが「個体ごとのばらつき（標準偏差）」を大きく跳ね上げます。このばらつきの大きさが、そもそも要求サンプル数を引き上げている原因です。
                    * **③ 「面積あたりの実力」との乖離:** 農業の現場で知りたいのは「1株のエリート」ではなく、「群落（畑全体）としての面積あたり収量」です。単株のデータだけでは、現場のリアルな評価とズレが生じやすくなります。
                    """)

                elif unit == "区画" and n_ceil > 5:
                    st.warning(f"""
                    **【警告】1処理あたり {n_ceil} 区画（反復）** という数は、通常の野外圃場試験（一般的に3〜5反復）としては**非常に大規模**であり、場所や労力の確保が困難になる可能性が高いです。
                    """)
                    if plants_per_plot is not None and plants_per_plot < 10:
                        st.info(
                            f"💡 **1区画あたりの株数（現在: {plants_per_plot} 株）を増やすことも有効です。**"
                            "株数が増えると個体差が平均化されてCVが下がり、必要反復数が減る傾向があります。"
                            "例えば1区画10〜20株程度を目安に、標準偏差の入力値を見直してみてください。"
                        )

                st.markdown("""
                ### 🌾 なぜ単年の野外試験で有意差を出すのは難しいのか？
                農業の野外試験では、土壌のムラ、日当たり、気象条件など**「コントロールできない環境ノイズ」**が非常に大きくなります。
                単純な統計手法（t検定や一般的な分散分析）では、これらのノイズがすべて「誤差」として扱われるため、**「本当に新しい技術の効果なのか、たまたま環境が良かっただけなのか」を見分けるのが極めて困難**になります。結果として、偶然を排除するために非現実的な反復数が要求されてしまうのです。

                ### 💡 現実的な解決策：複数区画の合算、年次反復、そして高度な解析 (GLMM)
                無理に大量の単株調査や巨大な試験を行うのではなく、現代の農学研究では以下のアプローチが推奨されています。

                1. **「区画」単位での評価（単株のノイズを吸収する）**
                   1株ずつ測るのではなく、例えば「10株をまとめて1つの区画（プロット）」として収穫・計量します。これにより個体ごとのミクロなノイズが平均化され、データのばらつき（標準偏差）がグッと小さくなり、少ない反復数でも有意差が出やすくなります。
                2. **年次・場所を分ける（時間的・空間的な反復）**
                   1年で無理な反復数を行うのではなく、「毎年3反復を3年間」行うことで、年ごとの気象変動に対する「技術の安定性（再現性）」を評価します。
                3. **一般化線形混合モデル (GLMM) による解析**
                   年次ごとの気象の違いや、ブロックごとの土壌のムラを**「変量効果（Random Effect）」**として統計モデルに組み込みます。巨大な環境ノイズを単なる誤差から切り離すことで、「純粋な処理の効果」だけを高感度で抽出できます。

                4. **要因配置法（Factorial Design）の活用**
                   「品種」と「施肥量」など複数の要因を同時に試験する場合、各要因の主効果は**全区画がそれぞれの反復として機能**します（例：2要因 A×B の場合、Factor Aの有効反復数はB全水準分だけ増える）。単一要因の試験より少ない反復数で高い推定精度が得られます。ただし区画の総数は増えるため、畑の面積確保が前提です。また「水準を増やすだけ（1要因のまま）」では必要反復数は減らない点に注意してください。

                **👉 この計算機で「現実的ではない反復数」が出た場合こそが、単純な試験設計の限界であり、「データの取り方の見直し」や「年次反復とGLMMを前提とした高度な実験計画」へステップアップするサインなのです。**
                """)

# ==========================================
# タブ2: 圃場配置マップ生成
# ==========================================
with tab2:
    st.header("🗺️ 圃場配置マップ生成 (実験計画法)")
    st.markdown("場所のムラ（日当たり、水はけ等）による偏りを防ぐため、科学的に妥当な**「ランダム配置マップ」**を作成します。")

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

        elif "完全無作為化法" in design_type:
            st.info("💡 **完全無作為化法の特徴**: 環境のムラが全くない場所で使います。どこに何が来るか完全にランダムなため、屋外の畑で使うと特定の処理が偏って配置される危険があります。")
            reps = st.number_input("1処理あたりの反復数", min_value=2, value=4, step=1)
            cols = st.number_input("表示する圃場の列数", min_value=1, value=n_t, step=1)

        elif "ラテン方格法" in design_type:
            st.info(f"💡 **ラテン方格法の特徴**: 「処理数 = 行数 = 列数」という厳しい制約がありますが、縦方向のムラと横方向のムラを両方とも相殺できる強力な手法です。（現在: {n_t} × {n_t}区画になります）")
            blocks = n_t
            if n_t < 3:
                st.warning("🚨 ラテン方格法は処理数を3つ以上にしてください。")

        generate_btn = st.button("配置マップを生成", type="primary", use_container_width=True)

        if generate_btn:
            # ★修正: n_t < 2 の明示的チェック
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
                        df_layout = pd.DataFrame(layout,
                                                 index=[f"ブロック{i+1}" for i in range(int(blocks))],
                                                 columns=[f"区画{i+1}" for i in range(n_t)])

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
                        # ★修正: 処理ラベルも置換してランダマイズ
                        base = np.arange(n_t)
                        square = np.array([np.roll(base, -i) for i in range(n_t)])
                        np.random.shuffle(square)
                        square = square[:, np.random.permutation(n_t)]
                        t_perm = np.random.permutation(n_t)   # ★追加
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

            # ★修正: ListedColormap + BoundaryNorm で色ズレを防止
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

            fig, ax = plt.subplots(figsize=(8, df_num.shape[0] * 1.0))
            sns.heatmap(df_num, annot=df_layout, fmt="", ax=ax,
                        cmap=cmap, norm=norm, cbar=False,
                        linewidths=2, linecolor='white',
                        annot_kws={"size": 14, "weight": "bold"})
            plt.yticks(rotation=0)
            ax.xaxis.tick_top()
            st.pyplot(fig)

            csv = df_layout.to_csv().encode('utf-8-sig')
            st.download_button("📥 この配置図をCSVでダウンロード", csv, "field_layout.csv", "text/csv", use_container_width=True)