from typing import Annotated
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from argparse import Namespace
import uvicorn
import onnxruntime_genai as og


class Ph3Model:
    def __create_model(self, args):
        self.__model = og.Model(f'{args.model}')
        self.__tokenizer = og.Tokenizer(self.__model)
        self.__tokenizer_stream = self.__tokenizer.create_stream()

        self.__search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
        if 'max_length' not in self.__search_options:
            self.__search_options['max_length'] = 2048

    def get_output(self, system_input, user_input):
        chat_template = '<|system|>\n{system}<|end|><|user|>\n{input}<|end|>\n<|assistant|>'
        prompt = f'{chat_template.format(system=system_input,input=user_input)}'
        input_tokens = self.__tokenizer.encode(prompt)

        params = og.GeneratorParams(self.__model)
        params.set_search_options(**self.__search_options)
        params.input_ids = input_tokens

        generator = og.Generator(self.__model, params)

        output_str = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            output_str += self.__tokenizer_stream.decode(new_token)
        
        del generator
        return(output_str)

    def __init__(self):
        """
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
        parser.add_argument('-m', '--model', type=str, default="cuda/cuda-int4-rtn-block-32", help='Onnx model folder path (must contain config.json and model.onnx)')
        parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
        parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
        parser.add_argument('-ds', '--do_sample', action='store_true', default=False, help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
        parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
        parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
        parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
        parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
        parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
        parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
        """
        args = Namespace(do_sample=False, verbose=False, timings=False, model='cuda/cuda-int4-rtn-block-32')
        self.__create_model(args)


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Start creating model")
ph3_model = Ph3Model()
print("Done creating model")


@app.post("/phi3/")
async def process_input(user_input: Annotated[str, Form()]):
    try:
        system_input = "Give weather information"
        output = ph3_model.get_output(system_input, user_input)

        return {"status": True, "message": output}
    except:
        return {"status": False, "message": "Fail to get phi3 output data, please try again."}
    

if __name__ == "__main__":
    uvicorn.run(app, port=6000)
