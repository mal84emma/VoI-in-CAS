"""Plot LP over-optimism results."""

import json

from matplotlib import pyplot as plt



if __name__ == '__main__':

    with open('det_LP_optimism_results.json','r') as json_file:
        results = json.load(json_file)
    
    print(results)

    plt.plot(list(results.keys()),[item['optimism']['perc'] for item in results.values()],'ko')
    #plt.plot(list(results.keys()),[item['optimism']['perc'] for item in results.values()],'k-',alpha=0.25)
    plt.xlim(-0.5,len(results)-0.5)
    plt.ylim(0)
    plt.xlabel("Number of buildings")
    plt.ylabel("LP Objective Over-Optimism (%)")
    plt.grid(True,'major',alpha=0.5,linestyle='--')
    plt.show()