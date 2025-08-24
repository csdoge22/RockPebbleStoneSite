import pandas as pd


def main():
    df = pd.read_csv("./rps_dataset.csv")["text"]
    tasks = set()
    duplicates = set()
    
    for i in range(df.shape[0]):
        if(df.iloc[i] in tasks and df.iloc[i] in duplicates):
            continue
        elif(df.iloc[i] in tasks):
            duplicates.add(df.iloc[i])
        else:
            tasks.add(df.iloc[i])
    
    print(duplicates)
        
        

if __name__=="__main__":
    main()