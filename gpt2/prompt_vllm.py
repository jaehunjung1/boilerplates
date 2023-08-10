import random
import logging

import ipdb
from vllm import LLM, SamplingParams
from nltk.tokenize import sent_tokenize


if __name__ == "__main__":
    logging.getLogger("vllm").setLevel(logging.WARNING)

    llm = LLM(model="gpt2-xl", seed=random.randint(0, 10000))

    prefix_sampling_params = SamplingParams(n=20, temperature=1.0, top_p=0.9, max_tokens=150)
    sent_sampling_params = SamplingParams(n=20, temperature=0.7, top_p=0.9, max_tokens=50)

    prefix = "New York (CNN) --"
    raw_outputs = llm.generate([prefix], prefix_sampling_params, use_tqdm=False)[0].outputs

    for raw_output in raw_outputs:
        generated_paragraph = " ".join(sent_tokenize(raw_output.text)[:-1])
        prefix_paragraph = prefix + generated_paragraph

        prompts = [prefix_paragraph + end for end in [" Unsurprisingly,", " Surprisingly,"]]

        raw_outputs = llm.generate(prompts, sent_sampling_params, use_tqdm=False)
        outputs1 = raw_outputs[0].outputs
        outputs2 = raw_outputs[1].outputs
        ipdb.set_trace()
        pass
    pass


