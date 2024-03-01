if [ ! -z "$3" ]; then
    python /home/evrun/evorobot-integrated/src/bin/es.py -f ./environments/$1 -s $2 -l $3
else
    python /home/evrun/evorobot-integrated/src/bin/es.py -f ./environments/$1 -s $2
fi