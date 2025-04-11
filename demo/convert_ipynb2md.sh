check_pip(){
    pkg_name=$1
    ret=$(pip show ${pkg_name} 2>&1 | grep "not found"  | wc -l)
    if [[ $ret == 1 ]];then
        pip install ${pkg_name}
    fi

    echo "installed [${pkg_name}]"
}

check_pip pandoc
check_pip nbconvert

jupyter nbconvert --to markdown $1