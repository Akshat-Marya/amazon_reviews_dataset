import sys, gzip, json, os, csv
import numpy as np
import itertools, random


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def get_data(filename, count,randomise_coeff=False):
    
    def get_items(indexes,count):
        helpful = []
        sentences = []
        nth_count = 0
        h_count = 0
        limit=count//2
        for index in indexes:
                print(f'Loading dataset:{round(((nth_count+h_count)/count)*100,2)}%',end='\r\r',flush=True)
                item= next(itertools.islice(parse(filename), index, None))
                # get h value
                try:
                    h_val =  int(item['vote'].replace(',', ''))
                except:
                    h_val=0

                if h_val<=0 and nth_count<limit:
                    nth_count += 1
                    helpful.append(h_val)
                    try:
                        sentences.append(item['reviewText'])
                    except:
                        sentences.append('')
                elif h_val>=0 and h_count<limit:
                    h_count += 1
                    helpful.append(h_val)
                    try:
                        sentences.append(item['reviewText'])
                    except:
                        sentences.append('')
                else:
                    continue

                
                if h_count == limit and nth_count==limit:
                    break
        return helpful, sentences

    if randomise_coeff:
        try:
            indexes=sorted(random.sample(range(randomise_coeff), count))
        except ValueError as e:
            print('Sample larger than population')
            exit()
        # get data
        helpful,sentences=get_items(indexes, count)
    else:
        helpful = []
        sentences = []
        nth_count = 0
        h_count = 0
        limit=count//2
        for i,item in enumerate(parse(filename)):
                print(f'Loading dataset:{round(((nth_count+h_count)/count)*100,2)}%',end='\r\r',flush=True)
                # get h value
                try:
                    h_val =  int(item['vote'].replace(',', ''))
                except:
                    h_val=0

                if h_val<=0 and nth_count<limit:
                    nth_count += 1
                    helpful.append(h_val)
                    try:
                        sentences.append(item['reviewText'])
                    except:
                        sentences.append('')
                elif h_val>=0 and h_count<limit:
                    h_count += 1
                    helpful.append(h_val)
                    try:
                        sentences.append(item['reviewText'])
                    except:
                        sentences.append('')
                else:
                    continue

                
                if h_count == limit and nth_count==limit:
                    break

    # print stats
    print(f"helpful||count:{len(helpful)},total:{sum(helpful)},mean:{sum(helpful)/len(helpful)}")

    return helpful, sentences