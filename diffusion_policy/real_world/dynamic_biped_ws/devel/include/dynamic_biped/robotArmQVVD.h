// Generated by gencpp from file dynamic_biped/robotArmQVVD.msg
// DO NOT EDIT!


#ifndef DYNAMIC_BIPED_MESSAGE_ROBOTARMQVVD_H
#define DYNAMIC_BIPED_MESSAGE_ROBOTARMQVVD_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace dynamic_biped
{
template <class ContainerAllocator>
struct robotArmQVVD_
{
  typedef robotArmQVVD_<ContainerAllocator> Type;

  robotArmQVVD_()
    : header()
    , q()
    , v()
    , vd()
    , tau()  {
    }
  robotArmQVVD_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , q(_alloc)
    , v(_alloc)
    , vd(_alloc)
    , tau(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _q_type;
  _q_type q;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _v_type;
  _v_type v;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _vd_type;
  _vd_type vd;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _tau_type;
  _tau_type tau;





  typedef boost::shared_ptr< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> const> ConstPtr;

}; // struct robotArmQVVD_

typedef ::dynamic_biped::robotArmQVVD_<std::allocator<void> > robotArmQVVD;

typedef boost::shared_ptr< ::dynamic_biped::robotArmQVVD > robotArmQVVDPtr;
typedef boost::shared_ptr< ::dynamic_biped::robotArmQVVD const> robotArmQVVDConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dynamic_biped::robotArmQVVD_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dynamic_biped::robotArmQVVD_<ContainerAllocator1> & lhs, const ::dynamic_biped::robotArmQVVD_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.q == rhs.q &&
    lhs.v == rhs.v &&
    lhs.vd == rhs.vd &&
    lhs.tau == rhs.tau;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dynamic_biped::robotArmQVVD_<ContainerAllocator1> & lhs, const ::dynamic_biped::robotArmQVVD_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dynamic_biped

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
{
  static const char* value()
  {
    return "3871141b674f003bc326e4d8da08f4ad";
  }

  static const char* value(const ::dynamic_biped::robotArmQVVD_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x3871141b674f003bULL;
  static const uint64_t static_value2 = 0xc326e4d8da08f4adULL;
};

template<class ContainerAllocator>
struct DataType< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dynamic_biped/robotArmQVVD";
  }

  static const char* value(const ::dynamic_biped::robotArmQVVD_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n"
"float64[] q\n"
"float64[] v\n"
"float64[] vd\n"
"float64[] tau\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
;
  }

  static const char* value(const ::dynamic_biped::robotArmQVVD_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.q);
      stream.next(m.v);
      stream.next(m.vd);
      stream.next(m.tau);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct robotArmQVVD_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dynamic_biped::robotArmQVVD_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dynamic_biped::robotArmQVVD_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "q[]" << std::endl;
    for (size_t i = 0; i < v.q.size(); ++i)
    {
      s << indent << "  q[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.q[i]);
    }
    s << indent << "v[]" << std::endl;
    for (size_t i = 0; i < v.v.size(); ++i)
    {
      s << indent << "  v[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.v[i]);
    }
    s << indent << "vd[]" << std::endl;
    for (size_t i = 0; i < v.vd.size(); ++i)
    {
      s << indent << "  vd[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.vd[i]);
    }
    s << indent << "tau[]" << std::endl;
    for (size_t i = 0; i < v.tau.size(); ++i)
    {
      s << indent << "  tau[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.tau[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // DYNAMIC_BIPED_MESSAGE_ROBOTARMQVVD_H