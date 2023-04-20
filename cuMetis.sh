input="graph.csv"

{
  read
  i=1
  while IFS=',' read -r  Name 
  do
    ./cuMetis graph/$Name.graph 8 >> graph/$Name\_part8.txt
 
    i=`expr $i + 1`
  done 
} < "$input"
