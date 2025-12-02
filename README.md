폴더 구조(2025.12.02 기준)  
  
<pre>
root/
├── modules/
│   ├── gate_structures/
│   ├── gate_voltages/
│   ├── potential_2DEG/
│   └── helper/
│       └── plot/
│
└── simulations/
    ├── qpc/
    └── qd/
</pre>  

구조의 큰 지향점
-미래에 수많은 버그, 기존 코드 수정(수정 편의성), 코드 파일 추가(확장성)가 있다고 가정 및 고려  
-module 하위 폴더 및 하위 파일들은 각자가 속한 폴더가 아닌 외부 폴더와 되도록 의존성이 없게 코드 작성  
  
폴더 규칙  
-첫글자는 소문자로 작성  
-띄어쓰기는 _ (underscore)로 기입  
-아무나 폴더 이름만 보고도 어떤 기능을 하는지 알 수 있어야 함  
  
파일 규칙  
-비슷한 기능을 하는 파일끼리 모을 것  
-실행 파일은 root/simulations/폴더이름/... 에 종류별로 배치  
  
실행 방법  
-root 폴더에서 run code 실행(폴더가 지저분해지니 테스트할때면 여기서 돌리고 완성된 파일은 적절한 하위 폴더로 이동)  
-terminal 을 root 경로에 위치시키고 터미널에 python -m simulations.폴더.파일 입력(예:  python -m simulations.qpc.splitgate)  



