import os
import shutil
import json
import traceback

def main():
    root = f'./static/run'
    all_exp = [f'{root}/{ename}' for ename in os.listdir(root)]

    for exp in all_exp:
        if not os.path.isdir(exp):
            print(f'Not a directory: {exp}')
            continue
        buf = os.listdir(exp)
        if any([name.endswith('.pth') for name in buf]):
            print(f'Ignore: {exp}')
            continue
        
        
        if len(buf) == 0:
            os.removedirs(exp)
            print(f'{exp} removed')
            continue

        if len(buf) <= 2:
            # os.remove(exp)
            shutil.rmtree(exp)
            print(f'{exp} removed')
            continue
        try:
            with open(f'{exp}/config.json', 'r') as jin:
                config = json.load(jin)
    
                if config['debug']:
                    shutil.rmtree(f'{exp}')
                    print(f'debug folder {exp} removed')
                    continue
        except Exception as e:
            print(f'{exp} Error')
            # print(traceback.format_exc())
        

if __name__ == "__main__":
    main()