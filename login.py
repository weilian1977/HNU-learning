import requests

HEAD={
"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0",
"Accept": "application/xml, text/xml, */*; q=0.01",
"Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
"Accept-Encoding": "gzip, deflate",
"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
"Faces-Request": "partial/ajax",
"X-Requested-With": "XMLHttpRequest",
"Content-Length": "1063",
"Origin": "http://10.2.24.33:8080",
"Connection": "keep-alive",
"Referer": "http://10.2.24.33:8080/byod/templatePage/20160930191146463/guestRegister.jsf",
"Cookie": "JSESSIONID=11EDFFB0DF87878741E356BDFFFF063D; oam.Flash.RENDERMAP.TOKEN=nwvp8si7g"}

DATA="javax.faces.partial.ajax=true&javax.faces.source=mainForm%3Aj_id_r&javax.faces.partial.execute=mainForm&javax.faces.partial.render=mainForm%3Aerror+mainForm%3AforResetPwd&mainForm%3Aj_id_r=mainForm%3Aj_id_r&mainForm%3AforResetPwd=&userName=&userPwd=&userDynamicPwd=&userDynamicPwdd=&mainForm%3AuserNameLogin=S2209W0725&mainForm%3AserviceSuffixLogin=&mainForm%3ApasswordLogin=MTIzcGFzc3dvcmQ%3D&mainForm%3AuserDynamicPwd=&mainForm%3AuserDynamicPwdd=&mainForm_SUBMIT=1&javax.faces.ViewState=N8QBoU0SYLMEO%2BJ0%2B0V4WpquoRG%2BCg9rDflycI7%2F6S5H9PnJi51MrnySvs%2BcLTRh%2FD6GLauBXgvznYufxagPydLjeeOal0ZTgGOto7TLJLSXuOsjLoNt2m8%2B3GfrnAXNkBbPX3koRZisM4%2BKHvU6zLPb%2FP%2BYT9WTucj992FPnDRYQgDeXRmYFjyDEAhSGiuC5xNi9GjiLqcbBWoqTzBLt8ijuLQAsPK%2BdqbIBrVP7mDqbCVSBPCZsY3YnnMEHKXOWXwj1qDkjDdfD0GJ1PxIvuwmPJGaNjIKlP2TbiVcFuw8xeMQU8ZPFVYR9r21Uve75EItHn2PzixypC2BUruMhYXeQ5y4X4AmoEs%2FMdtQ6R1d6GY7gA9Mr6%2BBk8Oe99oYt9jQwCvq6%2FboYMxuXHA8U%2BsUdBXPFrvcrTfEzScIaiBR718rD7Q9C21VPaucDr0ZyBTkoqiiS36iWMuy7fXstqmjpZosD89Pa7KmrYgHLQNlpJxA4ESfWhajko9iWYU61ttZU8wfU7T7oiekjJ7Sp%2BsjQUo%3D"

url = 'http://10.2.24.33:8080/byod/templatePage/20160930191146463/guestRegister.jsf'
myobj = {'somekey': 'somevalue'}

x = requests.post(url, headers=HEAD, data = DATA)
print(dir(x))
print(x.status_code) 
print(x.text) 
