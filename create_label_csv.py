from bs4 import BeautifulSoup



def create_label_csv(entry_path,imageName):
    "Creates a CSV file with 'imageName, img_name, latex-label (not normalized), entry.path' on each row"
    entry_path = entry_path[2:]

    img_path = entry_path.split("\\")[0]
    img_name = entry_path.split("\\")[1]

    
    f = open(img_path +"/"+ img_name,"r")
    file = f.read()
    soup = BeautifulSoup(file, "xml")
    latex_label = soup.find(type="truth").string
    returnString = imageName +","+img_name + "," + latex_label + "," + img_path + " \n"

    f.close()
    return returnString

#create_label_csv(filepath,img_name)