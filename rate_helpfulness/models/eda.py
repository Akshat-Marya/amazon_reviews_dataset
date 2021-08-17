from sklearn.feature_extraction.text import TfidfVectorizer
import sys, gzip, json, os, csv


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def write_to_file(obj):
    #create directory
    write_to_folder='./intermediate_results'
    if not os.path.exists(write_to_folder):
        os.mkdir(write_to_folder)

    #write stats
    obj_names=['overall','helpful','images']
    header="type,total,count,max,min,mean".split(',')
    with open(f'{write_to_folder}/stats.csv','w',newline='') as f:
        spamwriter = csv.writer(f, delimiter=',')
        spamwriter.writerow(header)
        for i,dic in enumerate(obj[:-1]):
                spamwriter.writerow([obj_names[i]]+[v for v in dic.values()])
    #write product sets
    with open(f'{write_to_folder}/product_names.csv','w',newline='') as f:
        spamwriter = csv.writer(f, delimiter=',')
        for k,v in obj[-1].items():
            spamwriter.writerow([k,v])

def get_stats(filename):

    def fetch_item(name,item):
        try:
            val=item[name]
        except Exception as e:
            val=None

        return val

    def update_dict(val,data):
        if val is not None and type(val)==list:
            data['total']+=len(val)
            data['count']+=1
            if len(val) > data['max']:
                data['max']=len(val)
            else:
                # if a vote is ever negetive
                if data['min']=='':
                    data['min']=len(val)
                if len(val)<data['min']:
                    data['min']=len(val)

        else:
            val=float(val)
            data['total']+=val
            data['count']+=1
            if val > data['max']:
                data['max']=val
            else:
                # if a vote is ever negetive
                if data['min']=='':
                    data['min']=val
                if val<data['min']:
                    data['min']=val
        return data
    
    overall={'total':0,'count':0,'max':0, 'min':''}
    helpful={'total':0,'count':0,'max':0, 'min':''}
    image={'total':0,'count':0,'max':0, 'min':''}    
    product_sets={}

    for i,item in enumerate(parse(filename)):
        for name in item:
            if name=='overall':
                val = fetch_item(name,item)
                overall=update_dict(float(val),overall)
            if name=='vote':
                val = fetch_item(name,item)
                helpful=update_dict(float(val.replace(',','')),helpful)
            if name=='image':
                val = fetch_item(name,item)
                image=update_dict(val,image)
            if name=='asin':
                val=fetch_item(name,item)
                if val not in product_sets:
                    product_sets[val]=1
                else:
                    product_sets[val]+=1

        #print running stats
        if i%50000==0:
            print(f'Reading data point:{i}|overall:{overall}|helpful:{helpful}',flush=True,end='\r\r')
    
    overall['mean']=overall['total']/overall['count']
    helpful['mean']=helpful['total']/helpful['count']
    image['mean']=image['total']/image['count']
    total_count=i+1

    print()
    print(f'total_count:{total_count}')

    return overall,helpful,image,product_sets,total_count


if __name__=='__main__':

    filename='All_Amazon_Review.json.gz'
    overall,helpful,image,product_sets,total_count=get_stats(filename)
    write_to_file([overall,helpful,image,product_sets])
    print('finished')
