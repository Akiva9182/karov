#!/usr/bin/env python3
"""
classify_words.py — Classify Hebrew words for קרוב! difficulty levels.

Usage:
    pip install requests tqdm
    python classify_words.py --api-key YOUR_GEMINI_KEY

Output: word_classifications.json
"""
import argparse, json, os, time, threading, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

EMBEDDINGS_FILE = 'embeddings.json'
OUTPUT_FILE = 'word_classifications.json'
BATCH_SIZE = 100
MAX_WORKERS = 5
API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'

CATEGORIES = [
    "אוכל", "טבע", "בעלי_חיים", "רגשות", "משפחה", "גוף", "בית", "עבודה",
    "חינוך", "ספורט", "מוזיקה", "טכנולוגיה", "תחבורה", "ביגוד", "צבעים",
    "מזג_אוויר", "צבא", "רפואה", "משפט", "כלכלה", "דת", "אמנות", "מדע",
    "פוליטיקה", "מזון", "כלים", "ריהוט", "זמן", "מקום", "תקשורת", "חברה",
    "אופי", "פעולות", "מצבים", "כללי"
]


def classify_batch(words, api_key):
    """Send a batch of words to Gemini for classification. Returns dict {word: {difficulty, category}}."""
    prompt = f"""סווג את המילים הבאות. לכל מילה תן:
1. רמת קושי: easy (מילה שכל ילד בן 8 מכיר), medium (מילה שמבוגר ממוצע מכיר), hard (מילה ספרותית/מקצועית/נדירה)
2. קטגוריה אחת מהרשימה: {', '.join(CATEGORIES)}

החזר JSON בלבד, בפורמט:
[{{"word": "כלב", "difficulty": "easy", "category": "בעלי_חיים"}}, ...]

המילים: {json.dumps(words, ensure_ascii=False)}"""

    resp = requests.post(
        f'{API_URL}?key={api_key}',
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    text = data['candidates'][0]['content']['parts'][0]['text']
    classified = json.loads(text)

    # Build result dict, keyed by word
    result = {}
    for item in classified:
        w = item.get('word', '')
        if w:
            result[w] = {
                'difficulty': item.get('difficulty', 'medium'),
                'category': item.get('category', 'כללי')
            }
    return result


def process_batch(batch, batch_num, total_batches, api_key):
    """Process a single batch with retries. Returns (classified_dict, missing_words)."""
    sent_words = set(batch)

    for attempt in range(3):
        try:
            result = classify_batch(batch, api_key)
            # Check for missing words
            returned_words = set(result.keys())
            missing = sent_words - returned_words
            return result, list(missing)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return {}, list(sent_words)  # all failed

    return {}, list(sent_words)


def main():
    parser = argparse.ArgumentParser(description='Classify Hebrew words')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Concurrent workers')
    args = parser.parse_args()

    # Load words
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        all_words = list(json.load(f).keys())
    print(f'Total words in embeddings: {len(all_words)}')

    # Load existing classifications (incremental)
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f'Found existing classifications: {len(existing)}')

    # Filter to unclassified words
    todo = [w for w in all_words if w not in existing]
    print(f'Words to classify: {len(todo)}')
    if not todo:
        print('All words already classified!')
        return

    # Split into batches
    batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    total_batches = len(batches)
    print(f'Batches: {total_batches} ({args.workers} workers)')

    results = dict(existing)
    retry_queue = []
    lock = threading.Lock()
    save_counter = 0

    # Progress bar
    pbar = tqdm(total=len(todo), desc="Classifying", unit="word") if tqdm else None

    def handle_result(classified, missing, batch_num):
        nonlocal save_counter
        with lock:
            results.update(classified)
            if missing:
                retry_queue.extend(missing)
            save_counter += 1
            if pbar:
                pbar.update(len(classified))
            # Save every 5 batches
            if save_counter % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=None)

    # === Pass 1: parallel processing ===
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, batch in enumerate(batches):
            future = executor.submit(process_batch, batch, i + 1, total_batches, args.api_key)
            futures[future] = i + 1
            time.sleep(0.2)  # stagger submissions

        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                classified, missing = future.result()
                handle_result(classified, missing, batch_num)
                if missing:
                    print(f'\n  Batch {batch_num}: {len(missing)} words missing from response')
            except Exception as e:
                print(f'\n  Batch {batch_num} error: {e}')

    if pbar:
        pbar.close()

    # Save after pass 1
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=None)

    # === Pass 2: retry missing words ===
    # Filter retry queue to only words still not classified
    retry_queue = [w for w in retry_queue if w not in results]
    if retry_queue:
        print(f'\nRetrying {len(retry_queue)} missing words...')
        retry_batches = [retry_queue[i:i+BATCH_SIZE] for i in range(0, len(retry_queue), BATCH_SIZE)]

        for i, batch in enumerate(retry_batches):
            classified, still_missing = process_batch(batch, i + 1, len(retry_batches), args.api_key)
            results.update(classified)
            print(f'  Retry batch {i+1}/{len(retry_batches)}: {len(classified)} classified, {len(still_missing)} still missing')
            time.sleep(0.2)

        # Final save
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=None)

    # Summary
    classified_count = len(results) - len(existing)
    diffs = {'easy': 0, 'medium': 0, 'hard': 0}
    cats = set()
    for v in results.values():
        d = v.get('difficulty', 'medium')
        diffs[d] = diffs.get(d, 0) + 1
        cats.add(v.get('category', 'כללי'))

    total_missing = len(all_words) - len(results)
    print(f'\nDone! Results saved to {OUTPUT_FILE}')
    print(f'  Classified this run: {classified_count}')
    print(f'  Total classified: {len(results)}/{len(all_words)}')
    print(f'  Easy: {diffs["easy"]} | Medium: {diffs["medium"]} | Hard: {diffs["hard"]}')
    print(f'  Categories: {len(cats)}')
    if total_missing > 0:
        print(f'  Still missing: {total_missing} (run again to retry)')


if __name__ == '__main__':
    main()
