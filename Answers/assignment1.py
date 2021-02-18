phone_dir = {}

def inp():
    print("\nEnter your Name : ")
    name = str(input()).lower()
    if( not name.isalpha()) :
        print("Error! Not a valid Name!!")
        return
    phone_num = str(input("Enter your Phone Number : "));
    if((not phone_num.isdigit() ) or len(phone_num) > 10) :
        print("Error! Not a valid Phone Number!!")
        return
    email = input("Enter you email id : ")
    if(email.count('@') != 1) :
        print("Error! Not a valid email address!!")
        return
    addr = input("Enter your address without new lines(Press Enter if you don't wish to specify) : ")
    phone_dir[name] = [phone_num,email,addr]

def out():
    print("\nUser Details :\n\n")
    count = 1
    for name,li in phone_dir.items():
        print(f"{count} Name : {name}")
        print(f"   Phone number : {li[0]}")
        print(f"   Email ID : {li[1]}")
        print(f"   Address : {li[2]}\n")
        count += 1

n = 0
while(n < 5) :
    n = int(input("How many users do you wish to enter (min 5) : "))

for i in range(n):
    inp()

out()


