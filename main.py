from service.rag import RAG
from asset.String import clean_json_output
import datetime
import json
import pprint
#from service.comment import LLM


def main():
    pretty_print = pprint.PrettyPrinter(indent=4)
    # try:
    start_time = datetime.datetime.now()
    chatbot = RAG()
    text = ('今天青蛙有幾隻?')
    response = chatbot.predict('ner', text)
    print('NER:', response)
    response = chatbot.predict('sql', response)
    # print('SQL:', response)
    response = clean_json_output(response)
    answer = json.loads(response)
    pretty_print.pprint(answer)


if __name__ == '__main__':
    main()
