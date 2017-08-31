@echo off

git rev-list --count HEAD > VERSION

set /p minor=<VERSION

echo 4.4.%minor% > VERSION

echo ##teamcity[buildNumber '4.4.%minor%']
