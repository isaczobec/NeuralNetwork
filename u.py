
#IF(K5<>0;$K$2&CHAR(10);"")&

funcstring = "="
nLetters = ord("w") - ord("i")

for i in range(nLetters):
    funcstring += 'IF(' + chr(ord("i")+i).upper() +  '5<>0;$' +chr(ord("i")+i).upper() +  '$2&", "&' + chr(ord("i")+i).upper() + '5*100&"%, "&XLOOKUP('+'$' +chr(ord("i")+i).upper() +  '$2'+';Table2[Roll];Table2[Timkostnad])&" kr/h"&CHAR(10);"")&'
    # funcstring += 'IF(' + chr(ord("i")+i).upper() +  '5<>0;$' +chr(ord("i")+i).upper() +  '$2&", "&' + chr(ord("i")+i).upper() + '5*100&"%"&CHAR(10);"")&'


print(funcstring)

# IF($G4<>0;$G4/5/$U4&"x "&$G$1&CHAR(10);"")



# funcstring = ""
# nLetters = ord("s") - ord("e")

# for i in range(nLetters):
#     funcstring += 'IF($' + chr(ord('e')+i).upper() + '4<>0;$' + chr(ord('e')+i).upper() + '4/5/$U4&"x "&$' + chr(ord('e')+i).upper() + '$1&CHAR(10);"")&'


# print(funcstring)


