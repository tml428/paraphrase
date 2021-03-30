from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from functools import reduce
import random
import math
import json

tokenizer = AutoTokenizer.from_pretrained("model/")
model = AutoModelForSeq2SeqLM.from_pretrained("model/")

def get_decoded_output(output):
    return tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def get_normalized_rouge_output(output):
    print(output)
    if  output[1] > 0.7 and output[1] < 0.9:
        return True
    return False

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def predict(original_sentences, method="beam", maxProb = 0.9, minProb = 0.7):
    paraphrased_content = []
    
    for sentence in original_sentences:
        # comment to suppress output
        # print('original_sentence: ' + sentence)
        
        text =  "paraphrase: " + sentence + " </s>"
        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

        if method == "top_k":
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=256,
                early_stopping=True,

                # this block include args to using top k/p samplings
                do_sample=True,
                top_k=120,
                top_p=0.7,
                temperature=0.7
            )
        else:
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=256,
                early_stopping=True,

                # this block include args to using beam search
                num_return_sequences=10,
                num_beams=50,
                num_beam_groups=50,
                diversity_penalty=0.4
            )

        decoded_ouputs = list(map(get_decoded_output, outputs))

        rouge_scores = []
        for predicted_output in decoded_ouputs:
            score = scorer.score(sentence, predicted_output)
            score = score['rougeL'][2]
            rouge_scores.append(score)

        combined_results = zip(decoded_ouputs, rouge_scores)
        candidates = list(filter(lambda result: get_normalized_rouge_output(result), combined_results))
        # print(candidates)

        if candidates:
            paraphrased_content.append(candidates[math.floor(random.random()*len(candidates))][0])
            # comment to suppress output
            # print('winner: ' + winner[0])
        else: 
            paraphrased_content.append(sentence)
            # comment to suppress output
            # print('sentence: '+ sentence)

    return (' '.join(paraphrased_content))


def paraphrase(event, context):
    try:
        body = json.loads(event['body'])
        paraphrased_content = predict(body['original_content'], body['method'], body['max_prob'], body['min_prob'])

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Access-Control-Allow-Headers': '*'
            },
            "body": json.dumps({'paraphrased_content': paraphrased_content})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Access-Control-Allow-Headers': '*'
            },
            "body": json.dumps({"error": repr(e)})
        }