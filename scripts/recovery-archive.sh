#! /bin/bash
if [ -z $1 ]; then
    echo "USAGE: $0 (some).json[.gz]"
    exit 1
fi

extension=${1##*.}
echo "DIRECTORY: $(dirname $1)"
echo "FILE     : $(basename $1)"
echo "EXTENSION: $extension"

case $extension in
    (json) echo -e "\t\tnull\n\t]\n}\n" >> $1 ;;
    (gz)
        mv $1 $1.old
        ( gzip -d $1.old -c ;  echo -e "\t\tnull\n\t]\n}\n" ) | gzip -9 - -c >$1
        rm $1.old
        ;;
esac
