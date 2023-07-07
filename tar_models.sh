shopt -s extglob
tar czf models.tar.gz model/!(_img|old)
rm -rf model/!(_img|old)