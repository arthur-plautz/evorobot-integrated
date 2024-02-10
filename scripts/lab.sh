source scripts/version.sh
export lab_repo=./lab/repository

# Copy file structure to laboratory repository
sudo rm -R $lab_repo/*
cp lab/.gitlab-ci.yml $lab_repo
cp -r environments $lab_repo
cp -r src $lab_repo
cp -r scripts $lab_repo
cp .dockerignore $lab_repo
cp .gitignore $lab_repo
cp Dockerfile $lab_repo
cp Makefile $lab_repo
cp requirements.txt $lab_repo

# Publish changes with new version
cd $lab_repo
git add .
git commit -m "V$version Changes"
git push
git tag $version
git push --tags