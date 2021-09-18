import pickle
event=input("please enter the name of the event-")

n =12
a = list(map(float,input("\nEnter the numbers : ").strip().split()))[:n]

if event=="hockey":
    model = pickle.load(open("hockey.sav", 'rb'))
    print(model.predict([a]))
if event=="football":
    model = pickle.load(open("football.sav", 'rb'))
    print(model.predict([a]))
if event=="swimming":
    model = pickle.load(open("swimming.sav", 'rb'))
    print(model.predict([a]))
if event=="wrestling":
    model = pickle.load(open("wrestling.sav", 'rb'))
    print(model.predict([a]))
if event=="gymnastics":
    model = pickle.load(open("gymnastics.sav", 'rb'))
    print(model.predict([a]))
if event=="badminton":
    model = pickle.load(open("badminton.sav", 'rb'))
    print(model.predict([a]))
if event=="basketball":
    model = pickle.load(open("basketball.sav", 'rb'))
    print(model.predict([a]))
if event=="Weightlifting":
    model = pickle.load(open("Weightlifting.sav", 'rb'))
    print(model.predict([a]))
if event=="cycling":
    model = pickle.load(open("cycling.sav", 'rb'))
    print(model.predict([a]))
if event=="shooting":
    model = pickle.load(open("shooting.sav", 'rb'))
    print(model.predict([a]))   