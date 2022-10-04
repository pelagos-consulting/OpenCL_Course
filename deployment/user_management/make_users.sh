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
        adduser $usr --disabled-password --gecos "" 
    fi

    adduser $usr video
    adduser $usr render

    #change password for user
    echo "$usr:$pass" >> $fname
    
done

# Batch change user passwords
chpasswd < $fname
