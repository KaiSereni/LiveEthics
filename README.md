# LiveEthics Data Aggregation Algorithm

This is a Python project developed by [LiveEthics](https://liveethics.org) that analyzes companies' moral and ethical character (eg. Apple, Google, Microsoft, Meta) in several different categories by calling the Gemini API to research and score the companies.

### Notes on the ethics, effectiveness, and environmental impact of AI use

While we consider this an extremely important cause, it's important to note that AI has many negative impacts and important considerations. While this is true, we beleive that this use case is appropriate for AI for the following reasons:

- Ethics: the biggest ethical issue with AI is its replacement of human jobs and the plagarism of human work. In this case, however, were it not for AI research, this project would have never happened at all, meaning that no human researchers' jobs have been replaced. LiveEthics commits to never use AI art or image generation. We also would like to acknowledge that we are using an AI model whose training data came from human writers, which is an issue, but this moral hazard is, in our opinion, offset by the fact that we are giving higher scores businesses that lobby for progressive politicians who will hopefully begin to pass legeslation preventing these copyright abuses sometime soon.
- Effectiveness: AI is not human, and cannot make moral judgements. Asking an AI model to evaluate how ethical a company is is like asking a fish its opinion on deserts. For this reason, we designed this system such that the judgement is as objective as possible by having a team of humans select categories, the scores of which speak to the ethics of a company. Having separate categories also allows users to weight these scores based on their own moral compass, which is unique for everyone.
- Environmental impact: AI has massive environmental impacts. These can mainly be separated into two categories: water usage and electricity generation. Google, the company behind the language model we use, has made a deal with a nuclear power plant to generate their data centers' power. We commend this effort greatly, as nuclear power is one of the cleanest, safest, and most efficient power sources on the planet (contrary to oil companies' narratives), and is extremely underutilized. For more information, we reccommend reading *The Power of Nuclear* by Marco Visscher or researching using a source you trust. It's also important to note that the evaluations only need to be run once and then can be stored in a traditional database, only being updated periodically. This uses many orders of magnitude less power over time than services like ChatGPT, which call AI models for every single user query. It's also important to note that this service supports companies with a lower environmental impact, further offsetting any damage.

## Setup

1. **Install Package**

   Install the package using:
   
   ```
   pip install git+https://github.com/KaiSereni/liveethics#egg=liveethics
   ```
2. **API Keys**

Create a [Google Gemini API](https://aistudio.google.com/api-keys) token and add it to a `GEMINI_KEY` property in your .env file
```env
GEMINI_KEY=yourkeyhere
```

3. Use the package

Check out [eval.py](tests/eval.py) in the `tests` directory for an example usage

### Can you code it better? [Branch this code](https://github.com/KaiSereni/liveethics/branches) on GitHub!
