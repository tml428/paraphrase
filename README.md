# dibs-fe-paraphrase

The main component of this directory is a python function that accepts an array of strings and outputs a paragraph of the paraphrased strings. The python function is set up so to be deployed as an image-based lambda funtion, versus zip packages deployed in `ferrum/lambda`, since it uses a fine-tuned machine learning model and python libraries that existing lambda layer. In addition to the code essential to the paraphrasing feature, this directory also includes a cloudformation template file and a a buildspec file. The cloudformation template serves to create the lambda function integrate it with ApiGateway on AWS. The buildspec file, on the other hand, is used by AWS Codebuild to run pre-build commands, such as updating the docker image for the lambda function, and instructs AWS Cloudformation to run the deployment process specified in the cloudformation `template.yaml`.

## development-mode

When developing the paraphrase function, please run it in a container. At root, run
`$ docker build -t <tag> . && docker run -p 9000:8080 <tag>:latest`

Example request

```
curl --location --request POST 'http://localhost:9000/2015-03-31/functions/function/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "body": {
        "original_content": [
            "An assortment of Heywood Wakefield table lamps is available on 1stDibs.",
            "The range of these distinct items — often made of Wood — can elevate any home.",
            "If you’re looking to add Heywood Wakefield’s lighting to your home soon, you can find versions on 1stDibs in Brown, gray and undefined."
        ],
        "variables": [
            {"variableKey":"Heywood Wakefield table lamps",
             "variableValue":"<a href='\''www.heywoodWakfield.com'\''>"}
        ],
    }
}'
```

## Implementation details

This directory utilizes python, docker, and an array of AWS products (Cloudformation, Codebuild, Lambda, ApiGateway, iAM) to create a cloud-based service for running the paraphrasing function.

### The paraphrasing Model

- Using a [T5 model](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) [fine-tuned](https://huggingface.co/ceshine/t5-paraphrase-quora-paws) on three datasets -- [Paraphrase Adversaries from Word Scrambling](https://github.com/google-research-datasets/paws), [Quora Dataset Release QuestionPairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs), and [Paraphrase Grouped Corpora](https://www.oxinabox.net/resources/paraphrase_grouped_corpora/), the python function takes an array of sentences and generate a set of paraphrased output prediction using beam search method.
- When evaluating how The set of predictions are filtered down by rouge-L score, a metrics that is based on the concept of Longest Common Subsequence, to ensure the outputs are not identical to the original sentence while remaining factual.

### Codebuild

- [CodeBuild](https://aws.amazon.com/codebuild/) is a fully managed continuous integration service. It looks at the buildspec file for instructions on what commands need to be run for a build and run them.
- To use Codebuild, we have to create a codebuild project. This action only needs to be done once. Once a codebuild project is set up, the build will run when the build condition we define is met. For any new codebuild project to be able to create/update resources, it requires an iAM role to be created with the corresponding permissions. The role and its attached policy with the required permissions needs to be created separately if it has not already been created.

### Cloudformation

- [Cloudformation](https://aws.amazon.com/cloudformation/) allows developers to treat infrastructure as code. It creates a stack that contains all the resources (including Lambda, ApiGateway and iAM) needed for the application. It is what the serverless framework, which `ferrum/lambda` uses, is based on. Cloudformation is used in this directory instead of Serverless because it gives the developers more control (and Kiril complained about serverless), although the declarations also need to be more involved. There was also consideration to use the AWS in-house framework called Serverless Application Model but it does not support `!ImportValue` which is an important feature that allows resources to integrate with previously exported resources. (See [issue](https://github.com/aws/serverless-application-model/issues/1470))
