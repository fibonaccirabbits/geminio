# add to git files < 10MB in size
for f in $(find . -type f -size -10M)
	do 
	echo git adding $f
	git add $f
done

# ask for a commit message
read -p 'You shall not pass, unless you have a message: ' message
echo done! go code some more...
git commit -m '$message'
