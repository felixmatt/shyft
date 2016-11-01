@echo off

git rev-list --count --all > VERSION

set /p minor=<VERSION

echo 4.0.%minor% > VERSION

echo ##teamcity[buildNumber '4.0.%minor%']
