#!/bin/bash
$@
while [ $? -ne 0 ]; do
    $@
done