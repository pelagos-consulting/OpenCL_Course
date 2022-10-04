#!/bin/bash

fname=users.csv
rm $fname

for n in $(seq -f "%02g" 1 99)
do
    pass=$(gpw 1)
    usr="lab$n"

    # Check if user exists
    if id -u "$usr" > /dev/null 1>&1
    then
        # User exists
        echo $usr
    else
        # Create the user and add to groups
        adduser $usr 
        adduser $usr video
        adduser $usr render
    fi

    #change password for user
    echo "$usr:$pass" >> $fname
    
done
