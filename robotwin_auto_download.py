#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


TASKS = [
    'adjust_bottle','beat_block_hammer','blocks_ranking_rgb','blocks_ranking_size','click_alarmclock','click_bell',
    'dump_bin_bigbin','grab_roller','handover_block','handover_mic','hanging_mug','lift_pot','move_can_pot',
    'move_pillbottle_pad','move_playingcard_away','move_stapler_pad','open_laptop','open_microwave',
    'pick_diverse_bottles','pick_dual_bottles','place_a2b_left','place_a2b_right','place_bread_basket',
    'place_bread_skillet','place_burger_fries','place_can_basket','place_cans_plasticbox','place_container_plate',
    'place_dual_shoes','place_empty_cup','place_fan','place_mouse_pad','place_object_basket','place_object_scale',
    'place_object_stand','place_phone_stand','place_shoe','press_stapler','put_bottles_dustbin','put_object_cabinet',
    'rotate_qrcode','scan_object','shake_bottle_horizontally','shake_bottle','stack_blocks_three','stack_blocks_two',
    'stack_bowls_three','stack_bowls_two','stamp_seal','turn_switch'
]

ROOT = Path('/data/alice/cjtest/lara-wm/data/robotwin/dataset')
STATE_DIR = Path('/data/alice/cjtest/robotwin_download_state')
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_JSON = STATE_DIR / 'status.json'
LOG_PATH = STATE_DIR / 'download.log'


def log(msg: str) -> None:
    timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    line = f'[{timestamp}] {msg}'
    print(line, flush=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def gather_status() -> dict:
    downloaded = 0
    total_size = 0
    failed = []
    task_rows = []
    for task in TASKS:
        task_dir = ROOT / task
        zip_path = task_dir / 'aloha-agilex_clean_50.zip'
        zip_ok = zip_path.exists() and zip_path.stat().st_size > 0
        extracted_dir = task_dir / 'extracted' / 'aloha-agilex_clean_50'
        extracted = extracted_dir.exists()
        if zip_ok:
            downloaded += 1
            total_size += zip_path.stat().st_size
        task_rows.append({
            'task': task,
            'zip_exists': zip_path.exists(),
            'zip_size': zip_path.stat().st_size if zip_path.exists() else 0,
            'extracted': extracted,
        })
    return {
        'downloaded_count': downloaded,
        'target_count': len(TASKS),
        'download_root': str(ROOT),
        'downloaded_total_bytes': total_size,
        'failed_tasks': failed,
        'resume_supported': True,
        'tasks': task_rows,
    }


def save_status(status: dict) -> None:
    STATE_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    log('robotwin auto download started')
    status = gather_status()
    save_status(status)

    for task in TASKS:
        task_dir = ROOT / task
        task_dir.mkdir(parents=True, exist_ok=True)
        zip_path = task_dir / 'aloha-agilex_clean_50.zip'
        if zip_path.exists() and zip_path.stat().st_size > 0:
            log(f'skip existing zip: {task} ({zip_path.stat().st_size} bytes)')
            continue

        filename = f'dataset/{task}/aloha-agilex_clean_50.zip'
        try:
            log(f'downloading: {filename}')
            cached = hf_hub_download(
                repo_id='TianxingChen/RoboTwin2.0',
                repo_type='dataset',
                filename=filename,
                resume_download=True,
            )
            shutil.copy2(cached, zip_path)
            log(f'downloaded: {task} -> {zip_path} ({zip_path.stat().st_size} bytes)')
        except Exception as exc:
            log(f'failed: {task} -> {exc}')

        status = gather_status()
        save_status(status)

    status = gather_status()
    save_status(status)
    log('robotwin auto download finished pass')


if __name__ == '__main__':
    main()
