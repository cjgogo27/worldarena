from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import TypedDict, cast


ROOT = Path("/data/alice/cjtest")
METRICS_PATH = ROOT / "demos/demo3_flow_generalization/outputs/metrics.json"
OUTPUT_PATH = ROOT / "flow_matching_generalization_study_slides_cn.html"
REPORT_PATH = ROOT / "report_flow_matching_study/index.html"


class SummaryBlock(TypedDict):
    hypothesis_a: str
    hypothesis_b: str


class ConfigBlock(TypedDict):
    n_steps: int
    batch_size_1d: int
    batch_size_2d: int
    epochs_1d: int
    epochs_2d: int
    lr: float
    hidden_small: int
    hidden_large: int
    train_size_1d: int
    holdout_size_1d: int
    train_size_2d: int
    holdout_size_2d: int
    eval_points_1d: int
    eval_points_2d: int
    summary_seeds: list[int]


class SpectralMetrics(TypedDict):
    train_endpoint_mse: float
    holdout_endpoint_mse: float
    generalization_gap: float
    spectrum_error: float
    high_freq_capture_ratio: float
    smoothness_ratio: float
    endpoint_l1: float


class RandomnessMetrics(TypedDict):
    holdout_chamfer_like: float
    memorization_ratio: float
    radial_profile_l1: float
    angular_spectrum_error: float
    final_radius_mean: float
    final_radius_std: float


class SpectralEntry(TypedDict):
    case: str
    model_kind: str
    seed: int
    metrics: SpectralMetrics
    final_loss: float


class RandomnessRun(TypedDict):
    case: None
    model_kind: None
    seed: int
    metrics: RandomnessMetrics
    final_loss: float


class RandomnessSummary(TypedDict):
    holdout_chamfer_mean: float
    holdout_chamfer_std: float
    memorization_ratio_mean: float
    memorization_ratio_std: float
    angular_spectrum_error_mean: float
    angular_spectrum_error_std: float


class ArtifactBlock(TypedDict):
    spectral_bias_summary: str
    randomness_summary: str
    vector_field_summary: str
    combined_summary: str
    animation_gif: str
    animation_mp4: str | None


class MetricsRoot(TypedDict):
    demo: str
    summary: SummaryBlock
    config: ConfigBlock
    one_d_spectral_bias: list[SpectralEntry]
    two_d_randomness_runs: list[RandomnessRun]
    two_d_randomness_summary: RandomnessSummary
    artifacts: ArtifactBlock


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def metric_row(label: str, value: str, note: str) -> str:
    return (
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{escape(value)}</td>"
        f"<td>{escape(note)}</td>"
        "</tr>"
    )


def main() -> None:
    metrics = cast(MetricsRoot, json.loads(METRICS_PATH.read_text(encoding="utf-8")))

    config = metrics["config"]
    spectral = metrics["one_d_spectral_bias"]
    randomness_runs = metrics["two_d_randomness_runs"]
    randomness_summary = metrics["two_d_randomness_summary"]
    artifacts = metrics["artifacts"]

    low_freq = next(item for item in spectral if item["case"] == "low_freq")
    high_freq_small = next(
        item for item in spectral if item["case"] == "high_freq" and item["model_kind"] == "mlp_small"
    )
    high_freq_fourier = next(
        item for item in spectral if item["case"] == "high_freq" and item["model_kind"] == "fourier"
    )

    mean_loss_2d = sum(run["final_loss"] for run in randomness_runs) / len(randomness_runs)
    max_chamfer_seed = max(randomness_runs, key=lambda run: run["metrics"]["holdout_chamfer_like"])
    min_chamfer_seed = min(randomness_runs, key=lambda run: run["metrics"]["holdout_chamfer_like"])

    spectral_rows = "\n".join(
        [
            metric_row(
                "1D 低频 + 小 MLP",
                f"loss {fmt(low_freq['final_loss'])} / holdout MSE {fmt(low_freq['metrics']['holdout_endpoint_mse'])}",
                f"频谱误差 {fmt(low_freq['metrics']['spectrum_error'])}；最易拟合。",
            ),
            metric_row(
                "1D 高频 + 小 MLP",
                f"loss {fmt(high_freq_small['final_loss'])} / holdout MSE {fmt(high_freq_small['metrics']['holdout_endpoint_mse'])}",
                f"频谱误差 {fmt(high_freq_small['metrics']['spectrum_error'])}；明显退化。",
            ),
            metric_row(
                "1D 高频 + Fourier MLP",
                f"loss {fmt(high_freq_fourier['final_loss'])} / holdout MSE {fmt(high_freq_fourier['metrics']['holdout_endpoint_mse'])}",
                f"频谱误差 {fmt(high_freq_fourier['metrics']['spectrum_error'])}；优化更容易，但端点几何仍难。",
            ),
        ]
    )

    seed_rows = "\n".join(
        [
            "".join(
                [
                    "<tr>",
                    f"<td>{run['seed']}</td>",
                    f"<td>{fmt(run['metrics']['holdout_chamfer_like'])}</td>",
                    f"<td>{fmt(run['metrics']['memorization_ratio'])}</td>",
                    f"<td>{fmt(run['metrics']['angular_spectrum_error'])}</td>",
                    f"<td>{fmt(run['final_loss'])}</td>",
                    "</tr>",
                ]
            )
            for run in randomness_runs
        ]
    )

    low_vs_high_ratio = high_freq_small["metrics"]["holdout_endpoint_mse"] / low_freq["metrics"]["holdout_endpoint_mse"]
    spectral_ratio = high_freq_small["metrics"]["spectrum_error"] / low_freq["metrics"]["spectrum_error"]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Flow Matching 泛化玩具研究｜中文幻灯片</title>
  <meta name="description" content="已完成 toy Flow Matching generalization study 的中文本地幻灯片。">
  <style>
    :root {{
      --bg: #f5f1ea;
      --surface: #fffdf9;
      --ink: #1f1c18;
      --muted: #6c665f;
      --accent: #7d2f2f;
      --accent-soft: #eadfda;
      --border: #d8cec6;
      --shadow: 0 18px 48px rgba(58, 38, 23, 0.10);
      --slide-w: min(92vw, 1280px);
      --slide-h: min(51.75vw, 720px);
    }}

    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--ink); }}
    body {{
      font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }}
    .deck {{ padding: 28px 24px 72px; }}
    .slide {{
      width: var(--slide-w);
      min-height: var(--slide-h);
      margin: 0 auto 28px;
      padding: 42px 48px 34px;
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: 18px;
      position: relative;
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
    }}
    .slide::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 8px;
      background: linear-gradient(180deg, var(--accent), #b36b5f);
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    h1, h2, h3 {{
      font-family: Georgia, "Times New Roman", serif;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0;
    }}
    h1 {{ font-size: 40px; line-height: 1.08; margin-bottom: 12px; }}
    h2 {{ font-size: 31px; margin-bottom: 14px; }}
    h3 {{ font-size: 20px; margin-bottom: 10px; }}
    p, li {{ font-size: 19px; color: var(--ink); }}
    .muted, .footer, figcaption, .small {{ color: var(--muted); }}
    .lede {{ font-size: 22px; max-width: 1000px; }}
    .two-col {{ display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 28px; align-items: start; }}
    .two-col.equal {{ grid-template-columns: 1fr 1fr; }}
    .three-col {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
    .panel {{ background: #fffaf4; border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 14px; }}
    .metric {{ background: var(--accent-soft); border-radius: 14px; padding: 14px 16px; }}
    .metric .k {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .metric .v {{ font-size: 28px; font-weight: 700; margin-top: 8px; color: var(--accent); }}
    ul {{ margin: 8px 0 0 0; padding-left: 22px; }}
    li {{ margin: 8px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 16px; }}
    th, td {{ border: 1px solid var(--border); padding: 10px 12px; vertical-align: top; }}
    th {{ background: var(--accent-soft); text-align: left; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 12px; border: 1px solid var(--border); background: #f8f4ef; }}
    figcaption {{ font-size: 14px; margin-top: 8px; line-height: 1.35; }}
    .hero-note {{ max-width: 980px; font-size: 18px; color: var(--muted); }}
    .footer {{ display: flex; justify-content: space-between; align-items: center; font-size: 14px; margin-top: 18px; }}
    .kbd {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid var(--border); border-bottom-width: 3px; border-radius: 8px; padding: 3px 8px; background: #fff; }}
    .controls {{
      position: fixed; right: 18px; bottom: 18px; display: flex; gap: 10px; align-items: center;
      background: rgba(255,253,249,0.92); border: 1px solid var(--border); border-radius: 999px; padding: 10px 14px; box-shadow: 0 10px 26px rgba(0,0,0,0.08);
      backdrop-filter: blur(8px);
    }}
    .controls button {{ border: 0; background: var(--accent); color: white; width: 34px; height: 34px; border-radius: 999px; font-size: 18px; cursor: pointer; }}
    .controls span {{ font-size: 14px; color: var(--muted); min-width: 58px; text-align: center; }}
    .slide.active {{ outline: 3px solid rgba(125,47,47,0.18); }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; word-break: break-all; }}
    .tight li {{ margin: 4px 0; font-size: 18px; }}
    .quote {{ font-size: 28px; line-height: 1.28; color: var(--accent); max-width: 1030px; margin-top: 10px; }}

    @media (max-width: 1100px) {{
      .slide {{ padding: 34px 34px 28px; min-height: auto; }}
      .two-col, .two-col.equal, .three-col, .metric-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 34px; }}
      h2 {{ font-size: 28px; }}
      p, li {{ font-size: 18px; }}
    }}

    @media print {{
      body {{ background: white; }}
      .deck {{ padding: 0; }}
      .slide {{ width: 100%; min-height: 100vh; margin: 0; border-radius: 0; border: 0; box-shadow: none; page-break-after: always; }}
      .controls {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="deck">
    <section class="slide" data-title="标题">
      <header>
        <div class="eyebrow">Toy Study · Flow Matching Generalization</div>
        <h1>Flow Matching 泛化玩具研究<br>中文幻灯片交付件</h1>
        <p class="lede">围绕两个核心问题：<strong>频谱偏置是否让模型以“平滑”方式看起来在泛化？</strong>以及<strong>训练随机性是否会改变 2D transport 的覆盖与记忆化代理指标？</strong></p>
      </header>
      <div>
        <div class="metric-grid">
          <div class="metric"><div class="k">1D 低频 holdout MSE</div><div class="v">{fmt(low_freq['metrics']['holdout_endpoint_mse'])}</div></div>
          <div class="metric"><div class="k">1D 高频 holdout MSE</div><div class="v">{fmt(high_freq_small['metrics']['holdout_endpoint_mse'])}</div></div>
          <div class="metric"><div class="k">2D Chamfer-like</div><div class="v">{fmt_pm(randomness_summary['holdout_chamfer_mean'], randomness_summary['holdout_chamfer_std'])}</div></div>
          <div class="metric"><div class="k">2D Angular Spectrum</div><div class="v">{fmt_pm(randomness_summary['angular_spectrum_error_mean'], randomness_summary['angular_spectrum_error_std'])}</div></div>
        </div>
        <p class="hero-note">格式结论：本机未发现 <span class="kbd">python-pptx</span>、<span class="kbd">PptxGenJS</span>、Marp/Reveal 等现成 PPT 生成库，因此采用<strong>无需额外依赖、可本地直接打开的单文件 HTML 幻灯片</strong>作为交付。所有数字均直接来自本地 <span class="path">{escape(rel(METRICS_PATH))}</span>，图像均引用本地真实产物。</p>
      </div>
      <div class="footer"><span>学术风格：克制、研究汇报导向</span><span>1 / 7</span></div>
    </section>

    <section class="slide" data-title="问题与假设">
      <header>
        <div class="eyebrow">Research Question</div>
        <h2>研究动机与待检验假设</h2>
      </header>
      <div class="two-col equal">
        <div class="panel">
          <h3>研究问题</h3>
          <p>Flow Matching 直接学习把噪声推向数据分布的向量场。这里关注的不是大规模 SOTA，而是：在可视化、可解释的玩具任务里，模型到底是在<strong>学结构</strong>，还是只是靠<strong>平滑偏置</strong>得到表面上的泛化？</p>
          <p class="quote">“如果高频目标显著更难，且不同 seed 又能改变 2D 覆盖，那么泛化就不只是‘学到了正确流形’这么简单。”</p>
        </div>
        <div class="panel">
          <h3>两条假设</h3>
          <ul>
            <li><strong>假设 A：频谱偏置 / 平滑效应。</strong> 小型普通 MLP 更容易拟合低频目标，会把高频梳状结构抹平；Fourier 特征能补回一部分高频细节。</li>
            <li><strong>假设 B：训练随机性。</strong> 在固定数据与结构下，初始化与 SGD 噪声仍会改变 2D flower transport 的覆盖、几何误差与记忆化代理指标。</li>
          </ul>
          <p class="small">英文原始概要来自 metrics.json：{escape(metrics['summary']['hypothesis_a'])} / {escape(metrics['summary']['hypothesis_b'])}</p>
        </div>
      </div>
      <div class="footer"><span>两个假设都在现有实验中被直接量化</span><span>2 / 7</span></div>
    </section>

    <section class="slide" data-title="实验设置">
      <header>
        <div class="eyebrow">Experimental Setup</div>
        <h2>实验设置：小规模、可视化、可复核</h2>
      </header>
      <div class="three-col">
        <div class="panel">
          <h3>1D 频谱偏置测试</h3>
          <ul class="tight">
            <li>从简单源线分布出发</li>
            <li>对比低频目标 vs 高频目标</li>
            <li>普通小 MLP 与 Fourier MLP 对照</li>
          </ul>
        </div>
        <div class="panel">
          <h3>2D flower 随机性测试</h3>
          <ul class="tight">
            <li>圆形点云 → flower 目标点云</li>
            <li>固定代码与超参，仅改变随机种子</li>
            <li>观察 coverage / memorization 代理指标</li>
          </ul>
        </div>
        <div class="panel">
          <h3>关键训练配置</h3>
          <ul class="tight">
            <li>ODE steps: {config['n_steps']}</li>
            <li>1D epochs: {config['epochs_1d']}</li>
            <li>2D epochs: {config['epochs_2d']}</li>
            <li>Batch size (1D / 2D): {config['batch_size_1d']} / {config['batch_size_2d']}</li>
            <li>Hidden width: {config['hidden_small']} / {config['hidden_large']}</li>
            <li>Train/Holdout (1D): {config['train_size_1d']} / {config['holdout_size_1d']}</li>
            <li>Train/Holdout (2D): {config['train_size_2d']} / {config['holdout_size_2d']}</li>
          </ul>
        </div>
      </div>
      <div class="panel" style="margin-top:18px;">
        <h3>评估指标</h3>
        <ul class="tight">
          <li><strong>1D：</strong>train / holdout endpoint MSE、spectrum error、high-frequency capture ratio、smoothness ratio、endpoint L1。</li>
          <li><strong>2D：</strong>holdout Chamfer-like、memorization ratio、radial profile L1、angular spectrum error、终态半径均值与方差。</li>
        </ul>
      </div>
      <div class="footer"><span>所有数值来自同一份本地 metrics.json</span><span>3 / 7</span></div>
    </section>

    <section class="slide" data-title="1D 结果">
      <header>
        <div class="eyebrow">Hypothesis A</div>
        <h2>1D 结果：频谱偏置证据最强</h2>
      </header>
      <div class="two-col">
        <figure>
          <img src="{escape(rel(Path(artifacts['spectral_bias_summary'])))}" alt="1D 频谱偏置总结图">
          <figcaption>本地真实图像：1D experiment outputs。低频任务明显更容易，高频任务误差更高；Fourier 特征改善优化，但没有根治端点几何偏差。</figcaption>
        </figure>
        <div>
          <div class="panel">
            <h3>定量结论</h3>
            <ul>
              <li>小 MLP 从低频切到高频后，holdout MSE 从 <strong>{fmt(low_freq['metrics']['holdout_endpoint_mse'])}</strong> 升到 <strong>{fmt(high_freq_small['metrics']['holdout_endpoint_mse'])}</strong>，约为 <strong>{fmt(low_vs_high_ratio, 2)}×</strong>。</li>
              <li>频谱误差从 <strong>{fmt(low_freq['metrics']['spectrum_error'])}</strong> 升到 <strong>{fmt(high_freq_small['metrics']['spectrum_error'])}</strong>，约为 <strong>{fmt(spectral_ratio, 2)}×</strong>。</li>
              <li>Fourier MLP 把最终 loss 压到 <strong>{fmt(high_freq_fourier['final_loss'])}</strong>，明显低于普通小 MLP 的 <strong>{fmt(high_freq_small['final_loss'])}</strong>；但 holdout MSE 仍有 <strong>{fmt(high_freq_fourier['metrics']['holdout_endpoint_mse'])}</strong>。</li>
            </ul>
          </div>
          <table style="margin-top:16px;">
            <tr><th>实验</th><th>数值</th><th>解释</th></tr>
            {spectral_rows}
          </table>
        </div>
      </div>
      <div class="footer"><span>结论：所谓“泛化”很可能部分来自平滑，而非完整恢复高频结构</span><span>4 / 7</span></div>
    </section>

    <section class="slide" data-title="2D 结果">
      <header>
        <div class="eyebrow">Hypothesis B</div>
        <h2>2D 结果：seed 随机性可见且可测</h2>
      </header>
      <div class="two-col">
        <figure>
          <img src="{escape(rel(Path(artifacts['randomness_summary'])))}" alt="2D 随机性总结图">
          <figcaption>四次独立训练的 flower morph 结果。即便代码和超参数完全相同，不同种子仍导致覆盖方式与终态几何出现差异。</figcaption>
        </figure>
        <div>
          <div class="panel">
            <h3>汇总指标</h3>
            <ul>
              <li>Chamfer-like：<strong>{fmt_pm(randomness_summary['holdout_chamfer_mean'], randomness_summary['holdout_chamfer_std'])}</strong></li>
              <li>Memorization ratio：<strong>{fmt_pm(randomness_summary['memorization_ratio_mean'], randomness_summary['memorization_ratio_std'])}</strong></li>
              <li>Angular spectrum error：<strong>{fmt_pm(randomness_summary['angular_spectrum_error_mean'], randomness_summary['angular_spectrum_error_std'])}</strong></li>
              <li>四个 seed 的平均最终 loss：<strong>{fmt(mean_loss_2d, 4)}</strong></li>
              <li>Chamfer-like 最好 seed：<strong>{min_chamfer_seed['seed']}</strong>（{fmt(min_chamfer_seed['metrics']['holdout_chamfer_like'])}）；最差 seed：<strong>{max_chamfer_seed['seed']}</strong>（{fmt(max_chamfer_seed['metrics']['holdout_chamfer_like'])}）。</li>
            </ul>
          </div>
          <table style="margin-top:16px;">
            <tr><th>Seed</th><th>Chamfer-like</th><th>Memorization ratio</th><th>Angular spectrum</th><th>Final loss</th></tr>
            {seed_rows}
          </table>
        </div>
      </div>
      <div class="footer"><span>结论：随机性不是视觉噪声，而是能改变几何指标的真实因素</span><span>5 / 7</span></div>
    </section>

    <section class="slide" data-title="向量场与动画">
      <header>
        <div class="eyebrow">Field Inspection</div>
        <h2>向量场切片与动态可视化</h2>
      </header>
      <div class="two-col equal">
        <figure>
          <img src="{escape(rel(Path(artifacts['vector_field_summary'])))}" alt="向量场总结图">
          <figcaption>向量场整体较平滑，但最终覆盖方式对优化轨迹敏感，因此“平滑”与“随机性”两个因素并非互斥，而是同时存在。</figcaption>
        </figure>
        <div>
          <figure>
            <img src="{escape(rel(Path(artifacts['animation_gif'])))}" alt="flower morph 动画">
            <figcaption>本地 GIF 动画（seed 0）。可直接在浏览器中播放，用于展示从 source circle 到目标 flower geometry 的连续 morph。</figcaption>
          </figure>
          <div class="panel" style="margin-top:18px;">
            <h3>如何读这页</h3>
            <ul class="tight">
              <li>左图回答“学到的速度场是什么样”。</li>
              <li>右图回答“样本如何随时间移动”。</li>
              <li>二者共同支持：训练得到的是一个平滑但并非唯一的 transport。</li>
            </ul>
          </div>
        </div>
      </div>
      <div class="footer"><span>图像与 GIF 均来自 demos/demo3_flow_generalization/outputs/</span><span>6 / 7</span></div>
    </section>

    <section class="slide" data-title="结论与交付">
      <header>
        <div class="eyebrow">Takeaways</div>
        <h2>结论与本地交付路径</h2>
      </header>
      <div class="two-col equal">
        <div class="panel">
          <h3>核心结论</h3>
          <ul>
            <li><strong>假设 A 获得最清晰支持：</strong>低频目标显著更易拟合，高频目标上普通 MLP 出现明显误差增幅。</li>
            <li><strong>Fourier 特征主要改善优化，不是万能修复：</strong>loss 降低，但端点误差与频谱误差仍高。</li>
            <li><strong>假设 B 也得到支持：</strong>seed 差异足以改变 2D coverage 与几何指标分布。</li>
            <li><strong>整体解释：</strong>toy Flow Matching 的“泛化”更像是“部分拟合 + 平滑偏置 + 训练随机性”的组合结果。</li>
          </ul>
        </div>
        <div class="panel">
          <h3>本地文件</h3>
          <ul class="tight">
            <li>幻灯片：<span class="path">{escape(rel(OUTPUT_PATH))}</span></li>
            <li>生成脚本：<span class="path">generate_flow_matching_slides_cn.py</span></li>
            <li>原报告：<span class="path">{escape(rel(REPORT_PATH))}</span></li>
            <li>指标 JSON：<span class="path">{escape(rel(METRICS_PATH))}</span></li>
            <li>主图：<span class="path">{escape(rel(Path(artifacts['combined_summary'])))}</span></li>
          </ul>
          <p class="small">打开方式：直接用浏览器打开 HTML；键盘 <span class="kbd">←</span> / <span class="kbd">→</span> 或底部按钮切换当前页高亮。打印到 PDF 时，每张 slide 自动分页。</p>
        </div>
      </div>
      <figure style="margin-top:18px;">
        <img src="{escape(rel(Path(artifacts['combined_summary'])))}" alt="combined summary 图">
        <figcaption>综合总览图：把 1D 频谱偏置与 2D seed 随机性并置，适合作为答辩或组会中的总览页。</figcaption>
      </figure>
      <div class="footer"><span>交付完成：单文件中文 HTML 幻灯片，直接本地可打开</span><span>7 / 7</span></div>
    </section>
  </main>

  <div class="controls" aria-label="slide controls">
    <button type="button" id="prev" aria-label="上一页">‹</button>
    <span id="counter">1 / 7</span>
    <button type="button" id="next" aria-label="下一页">›</button>
  </div>

  <script>
    const slides = [...document.querySelectorAll('.slide')];
    let current = 0;
    const counter = document.getElementById('counter');

    function focusSlide(index) {{
      current = Math.max(0, Math.min(index, slides.length - 1));
      slides.forEach((slide, i) => slide.classList.toggle('active', i === current));
      slides[current].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      counter.textContent = `${{current + 1}} / ${{slides.length}}`;
    }}

    document.getElementById('prev').addEventListener('click', () => focusSlide(current - 1));
    document.getElementById('next').addEventListener('click', () => focusSlide(current + 1));

    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') {{
        event.preventDefault();
        focusSlide(current + 1);
      }}
      if (event.key === 'ArrowLeft' || event.key === 'PageUp') {{
        event.preventDefault();
        focusSlide(current - 1);
      }}
      if (event.key === 'Home') {{
        event.preventDefault();
        focusSlide(0);
      }}
      if (event.key === 'End') {{
        event.preventDefault();
        focusSlide(slides.length - 1);
      }}
    }});

    const observer = new IntersectionObserver((entries) => {{
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      const idx = slides.indexOf(visible.target);
      if (idx >= 0) {{
        current = idx;
        slides.forEach((slide, i) => slide.classList.toggle('active', i === current));
        counter.textContent = `${{current + 1}} / ${{slides.length}}`;
      }}
    }}, {{ threshold: [0.35, 0.6, 0.85] }});

    slides.forEach((slide) => observer.observe(slide));
    focusSlide(0);
  </script>
</body>
</html>
"""

    _ = OUTPUT_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
