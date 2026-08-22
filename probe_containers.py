"""One-off diagnostic (not part of the pipeline): does the chosen model ingest
each container as raw bytes? Determines whether the mp4_aac64 arm is viable.

Deletable after Step 4 is settled.
"""
import base64
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.environ["FOUNDRY_BASE_URL"],
    api_key=os.environ["FOUNDRY_API_KEY"],
)
MODEL = os.environ["FOUNDRY_MODEL"]

m = pd.read_parquet("data/manifest.parquet")
item = m[m.item_id == m.item_id.iloc[0]].set_index("transform_id")
PROMPT = ("Listen to this audio. The speaker's emotion is one of: angry, "
          "disgusted, fearful, happy, neutral, sad. Respond with exactly one "
          "word from that list and nothing else.")

# (transform, path, format-label to declare in input_audio)
cases = [
    ("ref",           item.at["ref", "stim_path"],           "wav"),
    ("mp3_64",        item.at["mp3_64", "stim_path"],        "mp3"),
    ("mp4_aac64",     item.at["mp4_aac64", "stim_path"],     "mp4"),
    ("mp4_aac64",     item.at["mp4_aac64", "stim_path"],     "m4a"),
    ("mp4_aac64",     item.at["mp4_aac64", "stim_path"],     "aac"),
    ("roundtrip_wav", item.at["roundtrip_wav", "stim_path"], "wav"),
]


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def try_input_audio(path, fmt):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=16,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio",
             "input_audio": {"data": b64(path), "format": fmt}},
        ]}],
    )
    return resp.choices[0].message.content


def try_audio_url(path, mime):
    data_uri = f"data:{mime};base64,{b64(path)}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=16,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "audio_url", "audio_url": {"url": data_uri}},
        ]}],
    )
    return resp.choices[0].message.content


print(f"MODEL={MODEL}\n")
print("=== variant A: content type 'input_audio' with declared format ===")
for tid, path, fmt in cases:
    try:
        out = try_input_audio(path, fmt)
        print(f"  OK   {tid:14s} fmt={fmt:4s} -> {out!r}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:180]
        print(f"  FAIL {tid:14s} fmt={fmt:4s} -> {type(e).__name__}: {msg}")

print("\n=== variant B: content type 'audio_url' data-URI (mp4 only) ===")
for mime in ("audio/mp4", "audio/aac", "audio/x-m4a"):
    try:
        out = try_audio_url(item.at["mp4_aac64", "stim_path"], mime)
        print(f"  OK   mime={mime:14s} -> {out!r}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:180]
        print(f"  FAIL mime={mime:14s} -> {type(e).__name__}: {msg}")
