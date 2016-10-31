@echo off

git rev-list --count --all > REVISION

set /p minor=<REVISION

echo 4.0.%minor% > VERSION

echo ##teamcity[buildNumber '4.0.%minor%']
