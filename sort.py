def sortPlatePossibilities(possibleCharList):
    possibleCharList=sorted(possibleCharList,key=lambda subl: sum(c for _,c in subl),reverse=True)
    return possibleCharList

def sortCharacters(charList):
    charList.sort(key=lambda char:char.y1)
    charList.sort(key=lambda char:char.x1)
    # for i in charList:
    #     print( (i.x1,i.y1,i.x2,i.y2) )
    return charList

# def priporitySort(charList):
#     charList.sort(key=lambda char:char.x2)
#     grp=[]
#     chars=charList
#     while (len(chars)>0):
#         l=[i for i in char] 