from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from typing import Annotated
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from argparse import Namespace
from pymilvus import MilvusClient

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import convert_to_dataframe
from sentence_transformers import SentenceTransformer

import uvicorn
import os
import uuid
 


class Ph3Model:
    def __create_model(self, args):
        # self.__model = og.Model(f'{args.model}')
        # self.__tokenizer = og.Tokenizer(self.__model)
        # self.__tokenizer_stream = self.__tokenizer.create_stream()
        self.__model = AutoModelForCausalLM.from_pretrained( "./model", device_map="cuda",  torch_dtype="auto",  trust_remote_code=True, load_in_4bit=True)
        self.__tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 


    def get_output(self, system_input, user_input):
        chat_template = '<|system|>\n{system}<|end|><|user|>\n{input}<|end|>\n<|assistant|>'
        ## user_input='Question:'+user_input+'\n'+'Context:'+res_string
        prompt = f'{chat_template.format(system=system_input,input=user_input)}'

        pipe = pipeline( "text-generation", model=self.__model,     tokenizer=self.__tokenizer, ) 
        generation_args = {     "max_new_tokens": 500,  "return_full_text": False,  "temperature": 0.0,  "do_sample": False, } 

        output = pipe(prompt, **generation_args) 
        return(output[0]['generated_text'])
        

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
app_url = 'http://127.0.0.1:6001/'
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = MilvusClient("./milvus_demo.db")
client.create_collection(
    collection_name="demo_collection",
    dimension=384  # The vectors we will use in this demo has 384 dimensions
)


print("Start creating model")
ph3_model = Ph3Model()
search_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Done creating model")


@app.post("/phi3/")
async def process_input(user_input: Annotated[str, Form()]):
    try:
        system_input = "Give weather information"
        output = ph3_model.get_output(system_input, user_input)

        return {"status": True, "message": output}
    except:
        return {"status": False, "message": "Fail to get phi3 output data, please try again."}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if not os.path.exists('files'):
        os.makedirs('files')

    file_id = uuid.uuid4().hex
    file_location = f"files/{file_id + '.pdf'}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    elements = partition_pdf(file_location,content_type="application/pdf", strategy ="fast")
    chunks = chunk_by_title(elements,new_after_n_chars=1000,max_characters=1000)
    df_chunk=convert_to_dataframe(chunks)
    docs = df_chunk['text'].to_list()

    
    passage_embeddings = search_model.encode(docs)
    vectors = passage_embeddings
    data = [ {"id": i, "vector": vectors[i,:], "text": docs[i], "subject": file_id} for i in range(len(vectors)) ]
    client.insert(collection_name="demo_collection", data=data)

    return {"file_location": app_url + file_location, "file_id": file_id}

@app.post("/pdf_reader/")
async def process_input_for_pdf(user_input: Annotated[str, Form()], file_id: Annotated[str, Form()]):
    try:
        query_embedding = search_model.encode([user_input])
        vectors = query_embedding

        res = client.search(
            collection_name="demo_collection",
            data=[vectors[0].tolist()],
            filter=f"subject=='{file_id}'",
            limit=2,
            output_fields=["text", "subject"],
        )


        all_info  = ""
        for item in res[0]:
            all_info += (item['entity']['text'] + '\n\n')

        system_input = "Please answer user's question."
        user_prompt = f"Please answer the 'Question' only based on what 'Information' is given. DO NOT out of the scope from 'Information'. \nQuestion: {user_input} \nInformation: \n{all_info}"
        output = ph3_model.get_output(system_input, user_prompt)

        return {"status": True, "message": output}
    except:
        return {"status": False, "message": "Fail to get pdf read data, please try again."}



app.mount("/files", StaticFiles(directory="files"), name="files")

if __name__ == "__main__":
    uvicorn.run(app, port=6001)
