input="graph.csv"

{
  read
  i=1
  while IFS=',' read -r  Name 
  do
    ./mtx_to_graph mtx/$Name.mtx
 
    i=`expr $i + 1`
  done 
} < "$input"
