import heapq
import multiprocessing
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.corpus import cmudict
from collections import defaultdict
from functools import lru_cache
import torch
import numpy as np
from multiprocessing.managers import SyncManager

max_workers = multiprocessing.cpu_count()

class PriorityQueue:
    """Custom Priority Queue using heapq for use with Manager."""
    def __init__(self):
        self.queue = []
        self.lock = multiprocessing.Lock()  # For thread-safe access

    def get_queue(self):
        with self.lock:
            return self.queue

    def push(self, item):
        with self.lock:
            heapq.heappush(self.queue, item)

    def pop(self):
        with self.lock:
            if self.queue:
                return heapq.heappop(self.queue)
            return None

    def is_empty(self):
        with self.lock:
            return len(self.queue) == 0

    def size(self):
        with self.lock:
            return len(self.queue)

    def remove_worst(self, worst_item):
        with self.lock:
            if worst_item in self.queue:
                self.queue.remove(worst_item)
                heapq.heapify(self.queue)  # Re-heapify after removing an item

class MyManager(SyncManager):
    pass

MyManager.register("PriorityQueue", PriorityQueue)

def Manager():
    m = MyManager()
    m.start()
    return m

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

# Cache for scoring function
@lru_cache(maxsize=10000)
def score_sequence(sequence):
    input_ids = gpt2_tokenizer.encode(sequence, return_tensors="pt")
    with torch.no_grad():
        loss = gpt2_model(input_ids, labels=input_ids)[0]
    return loss.item()

# Phoneme-to-grapheme dictionary setup
phoneme_to_words = defaultdict(list)
cmudict_dict = cmudict.dict()
for word, phoneme_lists in cmudict_dict.items():
    for phoneme_seq in phoneme_lists:
        phoneme_tuple = tuple(phoneme_seq)
        phoneme_to_words[phoneme_tuple].append(word)

@lru_cache(maxsize=10000)
def phoneme_to_grapheme(phoneme_sequence):
    phoneme_sequence = tuple(phoneme_sequence)
    return phoneme_to_words.get(phoneme_sequence, None)

# Decoding function for multiprocessing with priority queue
def decode_worker(task_data, task_queue, task_manager):
    score, sentence, cursor, finished = task_data
    results = []
    max_window_size = 28

    for window_size in range(1, max_window_size):
        if cursor + window_size > len(phoneme_list):
            break
        phoneme_window = tuple(phoneme_list[cursor:cursor + window_size])
        grapheme_candidates = phoneme_to_grapheme(phoneme_window)
        if grapheme_candidates:
            for grapheme in grapheme_candidates:
                new_sentence = str(sentence + " " + grapheme).strip()
                score = score_sequence(new_sentence)
                results.append((score, new_sentence, cursor + window_size, cursor + window_size >= len(phoneme_list)))
                print(f"Sentence: {new_sentence}")
                print(f"Score: {score}\n")

    if not results:
        new_sentence = str(sentence + " ~").strip()
        score = score_sequence(new_sentence)
        results.append((score, new_sentence, cursor + 1, cursor + 1 >= len(phoneme_list)))
        print(f"Sentence: {new_sentence}")
        print(f"Score: {score}\n")

    # Put results in the priority queue
    for score, new_sentence, new_cursor, finished in results:
        if task_queue.size() >= max_workers:
            worst_sentence = max(task_queue.get_queue(), key=lambda x: x[0])
            worst_sentence_score = worst_sentence[0]
            if worst_sentence_score > score:
                task_queue.remove_worst(worst_sentence)
                task_queue.push((score, new_sentence, new_cursor, finished))
        else:
            task_queue.push((score, new_sentence, new_cursor, finished))

    return np.reshape(results, -1).tolist()

phoneme_list = ['F', 'EH1', 'DH', 'ER0', 'B', 'AY1', 'F', 'EH1', 'DH', 'ER0', 'D', 'IH0', 'T', 'EY1', 'L', 'B', 'AY1', 'D', 
                'IH0', 'T', 'EY1', 'L', 'AY1', 'W', 'ER1', 'K', 'T', 'AW1', 'T', 'AH0', 'N', 'D', 'AH0', 'CH', 'IY1', 'V', 'D', 
                'W', 'ER1', 'K', 'IH0', 'NG', 'IH0', 'N', 'F', 'R', 'AH1', 'N', 'T', 'AH1', 'V', 'DH', 'AH0', 'T', 'EH1', 'L']
##phoneme_list =  ['F', 'EH1', 'DH', 'ER0', 'B', 'AY1', 'F', 'EH1', 'DH', 'ER0', 'D', 'IH0', 'T', 'EY1', 'L', 'B', 'AY1', 'D', 
##                'IH0', 'T', 'EY1', 'L', 'AY1', 'W', 'ER1', 'K', 'T', 'AW1', 'T', 'AH0', 'N', 'D', 'AH0', 'CH', 'IY1', 'V', 'D', 
##                'W', 'ER1', 'K', 'IH0', 'NG', 'IH0', 'N', 'F', 'R', 'AH1', 'N', 'T', 'AH1', 'V', 'DH', 'AH0', 'T', 'EH1', 'L', 
#                'AH0', 'V', 'IH2', 'ZH', 'AH0', 'N', 'AH0', 'N', 'D', 'S', 'UW1', 'P', 'ER0', 'S', 'K', 'AH1', 'L', 'P', 'IY0', 
#                'HH', 'IY1', 'R', 'EH1', 'S', 'M', 'IY1', 'S', 'IH1', 'T', 'IH0', 'NG', 'N', 'EH1', 'K', 'S', 'T', 'T', 'UW1', 
#                'M', 'AY1', 'W', 'AY1', 'F', 'IH1', 'T', 'EH1', 'S', 'DH', 'AH0', 'OW1', 'N', 'L', 'IY0', 'P', 'IH1', 'K', 'CH', 
#                'ER0', 'AY1', 'T', 'UH1', 'K', 'AH1', 'V', 'DH', 'AH0', 'IH0', 'N', 'T', 'AY1', 'ER0', 'P', 'R', 'AA1', 'S', 'EH2', 
#                'S', 'AE1', 'Z', 'AY1', 'M', 'UW1', 'V', 'D', 'TH', 'R', 'UW1', 'AY1', 'AH0', 'CH', 'IY1', 'V', 'D']
