from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from random import randrange
import json
import base64
import re

tokenizer = AutoTokenizer.from_pretrained("model/")
model = AutoModelForSeq2SeqLM.from_pretrained("model/")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def get_decoded_output(output):
    return tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def get_normalized_rouge_output(output, maxProb, minProb):
    if minProb < output[1] and output[1] < maxProb:
        return True
    return False


def predict(original_sentences, variables=[]):
    maxProb = 0.9
    minProb = 0.5

    paraphrased_content = []
    for sentence in original_sentences:
        text = "paraphrase: " + sentence + " </s>"
        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=256,
            early_stopping=True,
            num_return_sequences=7,
            num_beams=14,
            num_beam_groups=14,
            diversity_penalty=0.4,
        )

        decoded_ouputs = list(map(get_decoded_output, outputs))

        rouge_scores = []
        for predicted_output in decoded_ouputs:
            score = scorer.score(sentence, predicted_output)
            score = score["rougeL"][2]
            rouge_scores.append(score)
        combined_results = zip(decoded_ouputs, rouge_scores)
        candidates = list(
            filter(lambda result: get_normalized_rouge_output(result, maxProb, minProb), combined_results)
        )

        if candidates:
            paraphrasedSentence = candidates[randrange(len(candidates))][0]
        else:
            paraphrasedSentence = sentence
        paraphrased_content.append(paraphrasedSentence)

    paraphrasedOutput = " ".join(paraphrased_content)
    for variable in variables:
        paraphrasedOutput = re.sub(
            "(" + variable["variableKey"] + ")",
            variable["variableValue"] + "\\1" + "</a>",
            paraphrasedOutput,
            0,
            re.IGNORECASE,
        )

    return {"paraphrased_content": paraphrasedOutput}


def constructResponse(statusCode, responseBody):
    return {
        "statusCode": statusCode,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Access-Control-Allow-Headers": "*",
        },
        "body": json.dumps(responseBody),
    }


def getPredictedResponse(requestBody):
    return predict(
        requestBody["original_content"],
        requestBody["variables"],
    )


def paraphrase(event, context):
    try:
        body = json.loads(base64.b64decode(event["body"]))
        paraphrased_content = getPredictedResponse(body)
        return constructResponse(200, paraphrased_content)
    except Exception as e:
        try:
            # body = json.loads(event["body"])
            # paraphrased_content = getPredictedResponse(body)
            paraphrased_content = getPredictedResponse(event["body"])
            return constructResponse(200, paraphrased_content)
        except Exception as e:
            return constructResponse(500, {"error": repr(e)})